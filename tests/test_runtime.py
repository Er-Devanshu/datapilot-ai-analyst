
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from datapilot.agents.runtime import (
    AgentResponse,
    DataPilotAgent,
)
from datapilot.analytics.executor import AnalyticsExecutionResult
from datapilot.analytics.planner import AnalyticsPlanningResult
from datapilot.analytics.planning import (
    AnalyticsPlan,
    AnalyticsPlanValidation,
)
from datapilot.visualization.models import VisualizationSpec
from datapilot.visualization.renderer import RenderedVisualization


def build_rendered_visualization() -> RenderedVisualization:
    """Create a small deterministic rendered visualization."""

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="revenue",
        title="Revenue by Region",
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=["West", "East"],
                y=[400, 200],
            )
        ]
    )

    return RenderedVisualization(
        specification=specification,
        figure=figure,
    )


def test_agent_response_from_state() -> None:
    """Public response should expose the important agent state."""

    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    rendered = build_rendered_visualization()

    state = {
        "question": "What is revenue by region?",
        "answer": "West generated the highest revenue.",
        "sql": (
            "SELECT region, SUM(revenue) "
            "FROM sales GROUP BY region"
        ),
        "result": dataframe,
        "row_count": 2,
        "analysis": "analysis",
        "visualization": rendered.specification,
        "rendered_visualization": rendered,
        "evidence": "evidence",
        "status": "answer_composed",
    }

    response = AgentResponse.from_state(state)

    assert response.question == (
        "What is revenue by region?"
    )
    assert response.answer == (
        "West generated the highest revenue."
    )
    assert response.sql is not None
    assert response.result is dataframe
    assert response.row_count == 2
    assert response.analysis == "analysis"
    assert response.visualization is rendered.specification
    assert response.rendered_visualization is rendered
    assert response.evidence == "evidence"
    assert response.status == "answer_composed"
    assert response.error is None


def test_agent_response_exposes_structured_analytics_state() -> None:
    """Public response should preserve structured analytics objects."""

    plan = AnalyticsPlan(
        question="What is revenue by region?",
        requests=(),
    )

    validation = AnalyticsPlanValidation(
        accepted_requests=(),
        skipped_analyses=(),
        fatal_errors=(),
    )

    planning_result = AnalyticsPlanningResult(
        plan=plan,
        validation=validation,
        prompt="planning prompt",
        model_name="test-model",
    )

    execution = AnalyticsExecutionResult(
        executions=(),
        skipped_analyses=(),
        fatal_errors=(),
    )

    state = {
        "question": "What is revenue by region?",
        "status": "answer_composed",
        "analytics_plan": plan,
        "analytics_plan_validation": validation,
        "analytics_planning_result": planning_result,
        "analytics_execution": execution,
        "analytics_planning_error": None,
    }

    response = AgentResponse.from_state(state)

    assert response.analytics_plan is plan
    assert response.analytics_plan_validation is validation
    assert response.analytics_planning_result is planning_result
    assert response.analytics_execution is execution
    assert response.analytics_planning_error is None


def test_agent_response_exposes_planning_failure() -> None:
    """Public response should expose a planning-generation failure."""

    plan = AnalyticsPlan(
        question="What is revenue by region?",
        requests=(),
    )

    validation = AnalyticsPlanValidation(
        accepted_requests=(),
        skipped_analyses=(),
        fatal_errors=(),
    )

    planning_result = AnalyticsPlanningResult(
        plan=plan,
        validation=validation,
        prompt="",
        model_name="unknown",
    )

    state = {
        "question": "What is revenue by region?",
        "status": "answer_composed",
        "analytics_plan": plan,
        "analytics_plan_validation": validation,
        "analytics_planning_result": planning_result,
        "analytics_planning_error": "Invalid JSON response.",
    }

    response = AgentResponse.from_state(state)

    assert response.analytics_plan is plan
    assert response.analytics_plan_validation is validation
    assert response.analytics_planning_result is planning_result
    assert response.analytics_planning_error == (
        "Invalid JSON response."
    )
    assert response.error is None


def test_agent_response_handles_missing_optional_state() -> None:
    """Optional response fields should have safe defaults."""

    state = {
        "question": "Show revenue.",
        "status": "started",
    }

    response = AgentResponse.from_state(state)

    assert response.question == "Show revenue."
    assert response.answer is None
    assert response.sql is None
    assert response.result is None
    assert response.row_count == 0
    assert response.analysis is None
    assert response.visualization is None
    assert response.rendered_visualization is None
    assert response.evidence is None
    assert response.status == "started"
    assert response.error is None

    assert response.analytics_plan is None
    assert response.analytics_plan_validation is None
    assert response.analytics_planning_result is None
    assert response.analytics_execution is None
    assert response.analytics_planning_error is None


def test_agent_response_constructor_remains_backward_compatible() -> None:
    """The original response constructor remains valid."""

    response = AgentResponse(
        question="Show revenue.",
        status="started",
        sql=None,
        result=None,
        row_count=0,
        analysis=None,
        visualization=None,
        evidence=None,
        answer=None,
    )

    assert response.analytics_plan is None
    assert response.analytics_plan_validation is None
    assert response.analytics_planning_result is None
    assert response.analytics_execution is None
    assert response.analytics_planning_error is None


def test_agent_rejects_empty_question() -> None:
    """The public runtime should reject empty questions."""

    agent = object.__new__(DataPilotAgent)

    with pytest.raises(ValueError):
        agent.ask("")


def test_agent_rejects_whitespace_question() -> None:
    """Whitespace-only questions should also be rejected."""

    agent = object.__new__(DataPilotAgent)

    with pytest.raises(ValueError):
        agent.ask("   ")


def test_agent_rejects_non_string_question() -> None:
    """The public runtime should reject non-string questions."""

    agent = object.__new__(DataPilotAgent)

    with pytest.raises(ValueError):
        agent.ask(None)


def test_agent_rejects_negative_repair_attempts() -> None:
    """Repair attempts cannot be negative."""

    with pytest.raises(ValueError):
        DataPilotAgent(
            llm=None,
            inspector=None,
            executor=None,
            max_repair_attempts=-1,
        )


def test_agent_rejects_invalid_trusted_change_columns() -> None:
    """Trusted change columns must be supplied as a set or frozenset."""

    with pytest.raises(ValueError):
        DataPilotAgent(
            llm=None,
            inspector=None,
            executor=None,
            change_value_columns=["revenue_change"],
        )


def test_agent_accepts_trusted_change_columns() -> None:
    """Trusted change columns should be retained immutably."""

    agent = DataPilotAgent(
        llm=None,
        inspector=None,
        executor=None,
        change_value_columns={"revenue_change"},
    )

    assert agent.change_value_columns == frozenset(
        {"revenue_change"}
    )
