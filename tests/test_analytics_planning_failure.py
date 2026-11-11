
from __future__ import annotations

import pandas as pd

from datapilot.agents.nodes import (
    analytics_execution_node,
    analytics_planning_node,
)
from datapilot.agents.state import DataPilotState
from datapilot.analytics.planner import AnalyticsPlanningResult


class InvalidPlanLLM:
    """LLM test double that returns SQL instead of analytics JSON."""

    def generate(self, prompt: str) -> str:
        return (
            "SELECT region, SUM(revenue) "
            "FROM sales GROUP BY region"
        )


def test_planning_failure_is_recorded_without_losing_core_state() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "North"],
            "revenue": [700.0, 200.0, 100.0],
        }
    )

    state: DataPilotState = {
        "question": "What is revenue by region?",
        "result": dataframe,
        "row_count": 3,
        "status": "analysis_complete",
    }

    planned = analytics_planning_node(
        state,
        llm=InvalidPlanLLM(),
    )

    assert planned["status"] == "analytics_plan_failed"
    assert planned["result"] is dataframe
    assert planned["row_count"] == 3
    assert planned["analytics_plan"].requests == ()
    assert isinstance(
        planned["analytics_planning_result"],
        AnalyticsPlanningResult,
    )
    assert planned["analytics_plan_validation"].fatal_errors[0].code == (
        "PLAN_GENERATION_FAILED"
    )
    assert planned["analytics_planning_error"] == (
        "The LLM response is not valid JSON."
    )


def test_execution_preserves_plan_generation_failure() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "North"],
            "revenue": [700.0, 200.0, 100.0],
        }
    )

    state: DataPilotState = {
        "question": "What is revenue by region?",
        "result": dataframe,
        "row_count": 3,
        "status": "analysis_complete",
    }

    planned = analytics_planning_node(
        state,
        llm=InvalidPlanLLM(),
    )

    executed = analytics_execution_node(planned)

    execution = executed["analytics_execution"]

    assert executed["status"] == "analytics_executed"
    assert executed["result"] is dataframe
    assert execution.executions == ()
    assert execution.skipped_analyses == ()
    assert len(execution.fatal_errors) == 1
    assert execution.fatal_errors[0].code == "PLAN_GENERATION_FAILED"
    assert executed["analytics_planning_error"] == (
        "The LLM response is not valid JSON."
    )
