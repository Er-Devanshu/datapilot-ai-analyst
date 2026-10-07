from __future__ import annotations

import pytest

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LLMResponse
from datapilot.sql.executor import SQLExecutor
from datapilot.sql.pipeline import SQLPipeline
from datapilot.sql.repair import SQLRepairer


class FakeLLM:
    """Deterministic LLM double for pipeline tests."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = responses
        self.index = 0

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        if self.index >= len(self.responses):
            raise RuntimeError(
                "FakeLLM has no more responses."
            )

        response = self.responses[self.index]
        self.index += 1

        return LLMResponse(
            text=response,
            model_name="fake-model",
        )


class StringResponseLLM:
    """Deterministic LLM double that returns a plain string."""

    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.response


def build_executor() -> SQLExecutor:
    data_source = DuckDBDataSource()

    data_source.execute(
        """
        CREATE TABLE fact_sales (
            net_amount DOUBLE
        )
        """
    )

    data_source.execute(
        """
        INSERT INTO fact_sales VALUES
            (100.0),
            (200.0),
            (300.0)
        """
    )

    return SQLExecutor(
        data_source=data_source,
    )


def test_sql_pipeline_executes_valid_sql() -> None:
    executor = build_executor()

    pipeline = SQLPipeline(
        llm=FakeLLM(
            [
                "SELECT SUM(net_amount) AS revenue FROM fact_sales"
            ]
        ),
        inspector=SchemaInspector(
            executor.data_source
        ),
        executor=executor,
    )

    result = pipeline.run(
        "What is the total revenue?"
    )

    assert result.repair_attempts == 0
    assert result.execution.row_count == 1
    assert result.execution.dataframe.iloc[0]["revenue"] == 600.0


def test_sql_pipeline_repairs_invalid_sql() -> None:
    executor = build_executor()

    pipeline = SQLPipeline(
        llm=FakeLLM(
            [
                "SELECT SUM(revenue) FROM fact_sales",
                "SELECT SUM(net_amount) AS revenue FROM fact_sales",
            ]
        ),
        inspector=SchemaInspector(
            executor.data_source
        ),
        executor=executor,
    )

    result = pipeline.run(
        "What is the total revenue?"
    )

    assert result.repair_attempts == 1
    assert result.execution.row_count == 1
    assert result.execution.dataframe.iloc[0]["revenue"] == 600.0


def test_sql_pipeline_fails_after_max_repair_attempts() -> None:
    executor = build_executor()

    pipeline = SQLPipeline(
        llm=FakeLLM(
            [
                "SELECT revenue FROM fact_sales",
                "SELECT revenue FROM fact_sales",
            ]
        ),
        inspector=SchemaInspector(
            executor.data_source
        ),
        executor=executor,
        max_repair_attempts=1,
    )

    with pytest.raises(
        RuntimeError,
        match="SQL validation failed after 1 repair attempts",
    ):
        pipeline.run(
            "What is the total revenue?"
        )


def test_sql_repairer_accepts_plain_string_llm_response() -> None:
    repairer = SQLRepairer(
        llm=StringResponseLLM(
            "SELECT SUM(net_amount) AS revenue FROM fact_sales"
        )
    )

    result = repairer.repair(
        question="What is the total revenue?",
        sql="SELECT SUM(revenue) FROM fact_sales",
        errors=(
            "Column 'revenue' is not allowed.",
        ),
        schema_context=(
            "fact_sales(net_amount DOUBLE)"
        ),
    )

    assert result.repaired_sql == (
        "SELECT SUM(net_amount) AS revenue FROM fact_sales"
    )
    assert result.model_name == "unknown"