
from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from datapilot.agents.runtime import AgentResponse, DataPilotAgent
from datapilot.analytics.answer_validation import AnswerValidationResult
from datapilot.data.schema import ColumnInfo, TableInfo


VALID_ANALYTICS_PLAN = json.dumps(
    {
        "requests": [
            {
                "request_id": "revenue_by_region",
                "analysis_type": "segmentation",
                "inputs": {
                    "dimension_column": "region",
                    "value_column": "revenue",
                },
                "depends_on": [],
            }
        ]
    }
)


@dataclass
class FakeLLM:
    """Deterministic LLM for runtime answer-validation tests."""

    answer_response: str = (
        "West generated the highest revenue with ₹700."
    )
    analytics_response: str = VALID_ANALYTICS_PLAN

    def generate(self, prompt: str) -> str:
        normalized_prompt = prompt.rstrip()

        if normalized_prompt.endswith(
            "RETURN THE JSON OBJECT NOW:"
        ):
            return self.analytics_response

        if normalized_prompt.endswith("ANSWER:"):
            return self.answer_response

        return (
            "SELECT region, SUM(revenue) AS revenue "
            "FROM sales "
            "GROUP BY region"
        )


class FakeSchemaInspector:
    """Return a minimal sales schema."""

    def inspect(self) -> tuple[TableInfo, ...]:
        return (
            TableInfo(
                name="sales",
                columns=(
                    ColumnInfo(
                        name="region",
                        data_type="VARCHAR",
                        nullable=False,
                    ),
                    ColumnInfo(
                        name="revenue",
                        data_type="DOUBLE",
                        nullable=False,
                    ),
                ),
            ),
        )


class FakeSQLExecutor:
    """Return deterministic sales data."""

    def execute(self, sql: str):
        assert "SELECT" in sql.upper()
        assert "sales" in sql.lower()

        dataframe = pd.DataFrame(
            {
                "region": ["West", "East", "North"],
                "revenue": [700.0, 200.0, 100.0],
            }
        )

        return type(
            "ExecutionResult",
            (),
            {
                "dataframe": dataframe,
                "row_count": len(dataframe),
            },
        )()


class FailingSQLExecutor:
    """Simulate a SQL execution failure."""

    def execute(self, sql: str):
        raise RuntimeError("Simulated SQL execution failure.")


def build_agent(
    llm: FakeLLM | None = None,
    executor=None,
) -> DataPilotAgent:
    return DataPilotAgent(
        llm=llm or FakeLLM(),
        inspector=FakeSchemaInspector(),
        executor=executor or FakeSQLExecutor(),
    )


def test_runtime_exposes_successful_answer_validation():
    response = build_agent().ask(
        "What is revenue by region?"
    )

    assert response.status == "answer_composed"
    assert response.answer is not None
    assert response.answer_validation is not None
    assert isinstance(
        response.answer_validation,
        AnswerValidationResult,
    )
    assert response.answer_validation.valid
    assert response.answer_validation.issues == ()


def test_runtime_exposes_unsupported_number_warning():
    original_answer = (
        "West generated the highest revenue with ₹700 "
        "and profit was ₹999."
    )

    response = build_agent(
        llm=FakeLLM(answer_response=original_answer),
    ).ask(
        "What is revenue by region?"
    )

    assert response.status == "answer_composed"
    assert response.answer == original_answer
    assert response.answer_validation is not None
    assert not response.answer_validation.valid

    assert any(
        issue.code == "unsupported_number"
        and issue.offending_text == "999"
        for issue in response.answer_validation.issues
    )


def test_runtime_preserves_answer_when_validation_has_warnings():
    original_answer = (
        "West generated the highest revenue with ₹700. "
        "The additional reported figure is ₹999."
    )

    response = build_agent(
        llm=FakeLLM(answer_response=original_answer),
    ).ask(
        "What is revenue by region?"
    )

    assert response.answer == original_answer
    assert response.status == "answer_composed"
    assert response.answer_validation is not None
    assert not response.answer_validation.valid

    assert any(
        issue.code == "unsupported_number"
        and issue.offending_text == "999"
        for issue in response.answer_validation.issues
    )


def test_runtime_validation_does_not_change_pipeline_status():
    response = build_agent(
        llm=FakeLLM(
            answer_response="Revenue was ₹999."
        ),
    ).ask(
        "What is revenue by region?"
    )

    assert response.status == "answer_composed"
    assert response.answer is not None
    assert response.answer_validation is not None
    assert not response.answer_validation.valid
    assert response.error is None


def test_failed_pipeline_has_no_answer_validation():
    response = build_agent(
        executor=FailingSQLExecutor(),
    ).ask(
        "What is revenue by region?"
    )

    assert response.status == "failed"
    assert response.answer is None
    assert response.answer_validation is None
    assert response.error == "Simulated SQL execution failure."


def test_agent_response_from_legacy_state_allows_missing_validation():
    response = AgentResponse.from_state(
        {
            "question": "What is revenue?",
            "status": "answer_composed",
            "answer": "Revenue is ₹600.",
        }
    )

    assert response.answer == "Revenue is ₹600."
    assert response.answer_validation is None
