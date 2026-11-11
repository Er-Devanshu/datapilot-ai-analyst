
from __future__ import annotations

import json

import pandas as pd
import pytest

from datapilot.agents.nodes import (
    analytics_execution_node,
    analytics_planning_node,
)
from datapilot.agents.state import DataPilotState
from datapilot.analytics.executor import AnalyticsExecutionResult
from datapilot.analytics.planning import (
    AnalyticsPlanValidation,
    PlanValidationIssue,
)
from datapilot.analytics.planner import AnalyticsPlanningResult
from datapilot.analytics.variance import VarianceResult


class FakeLLM:
    """Deterministic LLM test double."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["June", "June", "July", "July"],
            "revenue": [60.0, 40.0, 50.0, 35.0],
        }
    )


def variance_response() -> str:
    return json.dumps(
        {
            "requests": [
                {
                    "request_id": "revenue_variance",
                    "analysis_type": "variance",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "reference_period": "June",
                        "actual_period": "July",
                        "aggregation": "sum",
                    },
                    "depends_on": [],
                }
            ]
        }
    )


def test_analytics_planning_node_generates_and_validates_plan(
    dataframe: pd.DataFrame,
) -> None:
    llm = FakeLLM(variance_response())

    state: DataPilotState = {
        "question": "Explain the revenue variance from June to July.",
        "result": dataframe,
        "row_count": len(dataframe),
        "status": "analysis_complete",
    }

    result = analytics_planning_node(
        state,
        llm=llm,
    )

    planning_result = result["analytics_planning_result"]

    assert isinstance(
        planning_result,
        AnalyticsPlanningResult,
    )
    assert result["analytics_plan"] is planning_result.plan
    assert (
        result["analytics_plan_validation"]
        is planning_result.validation
    )
    assert len(planning_result.validation.accepted_requests) == 1
    assert planning_result.validation.skipped_analyses == ()
    assert result["question"] == state["question"]
    assert result["result"] is dataframe
    assert result["status"] == "analytics_plan_ready"
    assert llm.last_prompt is not None


def test_analytics_planning_node_passes_trusted_change_columns(
    dataframe: pd.DataFrame,
) -> None:
    dataframe["revenue_change"] = [-5.0, -4.0, -3.0, -3.0]

    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "revenue_contribution",
                    "analysis_type": "contribution",
                    "inputs": {
                        "dimension_column": "month",
                        "value_column": "revenue_change",
                    },
                    "depends_on": [],
                }
            ]
        }
    )

    state: DataPilotState = {
        "question": "Explain revenue change by month.",
        "result": dataframe,
    }

    result = analytics_planning_node(
        state,
        llm=FakeLLM(response),
        change_value_columns={"revenue_change"},
    )

    validation = result["analytics_plan_validation"]

    assert len(validation.accepted_requests) == 1
    assert validation.skipped_analyses == ()


def test_analytics_planning_node_does_not_trust_change_columns_by_default(
    dataframe: pd.DataFrame,
) -> None:
    dataframe["revenue_change"] = [-5.0, -4.0, -3.0, -3.0]

    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "revenue_contribution",
                    "analysis_type": "contribution",
                    "inputs": {
                        "dimension_column": "month",
                        "value_column": "revenue_change",
                    },
                    "depends_on": [],
                }
            ]
        }
    )

    state: DataPilotState = {
        "question": "Explain revenue change by month.",
        "result": dataframe,
    }

    result = analytics_planning_node(
        state,
        llm=FakeLLM(response),
    )

    validation = result["analytics_plan_validation"]

    assert validation.accepted_requests == ()
    assert len(validation.skipped_analyses) == 1
    assert "explicitly trusted" in (
        validation.skipped_analyses[0].reason
    )


def test_analytics_planning_node_rejects_missing_question(
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="question"):
        analytics_planning_node(
            {"result": dataframe},
            llm=FakeLLM(variance_response()),
        )


def test_analytics_planning_node_rejects_missing_result() -> None:
    with pytest.raises(ValueError, match="executed result"):
        analytics_planning_node(
            {"question": "Analyze revenue."},
            llm=FakeLLM(variance_response()),
        )


def test_analytics_planning_node_rejects_non_dataframe_result() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        analytics_planning_node(
            {
                "question": "Analyze revenue.",
                "result": [],
            },
            llm=FakeLLM(variance_response()),
        )


def test_analytics_execution_node_executes_validated_plan(
    dataframe: pd.DataFrame,
) -> None:
    llm = FakeLLM(variance_response())

    planned_state = analytics_planning_node(
        {
            "question": "Explain the revenue variance from June to July.",
            "result": dataframe,
        },
        llm=llm,
    )

    result = analytics_execution_node(planned_state)

    execution = result["analytics_execution"]

    assert isinstance(
        execution,
        AnalyticsExecutionResult,
    )
    assert len(execution.executions) == 1
    assert execution.skipped_analyses == ()
    assert execution.fatal_errors == ()

    variance = execution.get_result("revenue_variance")

    assert isinstance(variance, VarianceResult)
    assert variance.reference_value == 100.0
    assert variance.actual_value == 85.0
    assert variance.absolute_variance == -15.0

    assert result["question"] == planned_state["question"]
    assert result["result"] is dataframe
    assert result["analytics_plan"] is planned_state["analytics_plan"]
    assert result["status"] == "analytics_executed"


def test_analytics_execution_node_preserves_fatal_validation_errors(
    dataframe: pd.DataFrame,
) -> None:
    fatal = PlanValidationIssue(
        code="DEPENDENCY_CYCLE",
        message="Dependency cycle detected.",
    )

    state: DataPilotState = {
        "question": "Analyze revenue.",
        "result": dataframe,
        "analytics_plan_validation": AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=(),
            fatal_errors=(fatal,),
        ),
    }

    result = analytics_execution_node(state)

    execution = result["analytics_execution"]

    assert execution.executions == ()
    assert execution.fatal_errors == (fatal,)
    assert result["status"] == "analytics_executed"


def test_analytics_execution_node_rejects_missing_result() -> None:
    with pytest.raises(ValueError, match="executed result"):
        analytics_execution_node(
            {
                "question": "Analyze revenue.",
                "analytics_plan_validation": AnalyticsPlanValidation(
                    accepted_requests=(),
                    skipped_analyses=(),
                ),
            }
        )


def test_analytics_execution_node_rejects_missing_validation(
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="analytics plan validation",
    ):
        analytics_execution_node(
            {
                "question": "Analyze revenue.",
                "result": dataframe,
            }
        )


def test_analytics_execution_node_rejects_non_dataframe_result() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        analytics_execution_node(
            {
                "question": "Analyze revenue.",
                "result": [],
                "analytics_plan_validation": AnalyticsPlanValidation(
                    accepted_requests=(),
                    skipped_analyses=(),
                ),
            }
        )
