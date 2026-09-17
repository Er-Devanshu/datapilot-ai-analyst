from __future__ import annotations

import duckdb

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LLMResponse
from datapilot.sql.executor import SQLExecutor
from datapilot.sql.pipeline import SQLPipeline


class FakeLLM:
    """Deterministic LLM double for pipeline tests."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = responses
        self.index = 0

    def generate(self, prompt: str) -> LLMResponse:
        """Return the next predefined response."""

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


def build_test_database() -> DuckDBDataSource:
    """Create a small in-memory database for testing."""

    data_source = DuckDBDataSource(":memory:")

    data_source.connect()

    connection = data_source._connection

    if connection is None:
        raise RuntimeError(
            "DuckDB connection was not created."
        )

    connection.execute(
        """
        CREATE TABLE fact_sales (
            sales_key INTEGER,
            location_key INTEGER,
            net_amount DOUBLE
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE dim_location (
            location_key INTEGER,
            region VARCHAR
        )
        """
    )

    connection.execute(
        """
        INSERT INTO fact_sales VALUES
            (1, 1, 1000),
            (2, 1, 2000),
            (3, 2, 1500)
        """
    )

    connection.execute(
        """
        INSERT INTO dim_location VALUES
            (1, 'North'),
            (2, 'West')
        """
    )

    return data_source


def test_pipeline_executes_valid_sql() -> None:
    """A valid generated query should execute immediately."""

    database = build_test_database()

    llm = FakeLLM(
        [
            """
            SELECT
                d.region,
                SUM(s.net_amount) AS revenue
            FROM fact_sales AS s
            JOIN dim_location AS d
                ON s.location_key = d.location_key
            GROUP BY d.region
            ORDER BY revenue DESC
            """
        ]
    )

    inspector = SchemaInspector(database)
    executor = SQLExecutor(database)

    pipeline = SQLPipeline(
        llm=llm,
        inspector=inspector,
        executor=executor,
    )

    result = pipeline.run(
        "What is the total revenue by region?"
    )

    assert result.repair_attempts == 0
    assert result.execution.row_count == 2
    assert set(
        result.execution.dataframe["region"]
    ) == {"North", "West"}

    database.close()


def test_pipeline_repairs_invalid_sql() -> None:
    """An invalid query should be repaired before execution."""

    database = build_test_database()

    llm = FakeLLM(
        [
            """
            SELECT
                d.region,
                SUM(s.revenue) AS revenue
            FROM fact_sales AS s
            JOIN dim_location AS d
                ON s.location_key = d.location_key
            GROUP BY d.region
            """,
            """
            SELECT
                d.region,
                SUM(s.net_amount) AS revenue
            FROM fact_sales AS s
            JOIN dim_location AS d
                ON s.location_key = d.location_key
            GROUP BY d.region
            ORDER BY revenue DESC
            """,
        ]
    )

    inspector = SchemaInspector(database)
    executor = SQLExecutor(database)

    pipeline = SQLPipeline(
        llm=llm,
        inspector=inspector,
        executor=executor,
    )

    result = pipeline.run(
        "What is the total revenue by region?"
    )

    assert result.repair_attempts == 1
    assert result.execution.row_count == 2
    assert "net_amount" in result.sql
    assert "s.revenue" not in result.sql

    database.close()


def test_pipeline_stops_after_max_repair_attempts() -> None:
    """Repeatedly invalid SQL must eventually fail safely."""

    database = build_test_database()

    llm = FakeLLM(
        [
            """
            SELECT
                s.invalid_column
            FROM fact_sales AS s
            """,
            """
            SELECT
                s.invalid_column
            FROM fact_sales AS s
            """,
            """
            SELECT
                s.invalid_column
            FROM fact_sales AS s
            """,
        ]
    )

    inspector = SchemaInspector(database)
    executor = SQLExecutor(database)

    pipeline = SQLPipeline(
        llm=llm,
        inspector=inspector,
        executor=executor,
        max_repair_attempts=2,
    )

    try:
        pipeline.run(
            "What is the total revenue?"
        )
    except RuntimeError as exc:
        assert "validation failed" in str(exc).lower()
    else:
        raise AssertionError(
            "Pipeline should fail after maximum "
            "repair attempts."
        )

    database.close()