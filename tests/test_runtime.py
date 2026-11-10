
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from datapilot.agents.runtime import (
    AgentResponse,
    DataPilotAgent,
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


def test_agent_rejects_negative_repair_attempts() -> None:
    """Repair attempts cannot be negative."""

    with pytest.raises(ValueError):
        DataPilotAgent(
            llm=None,
            inspector=None,
            executor=None,
            max_repair_attempts=-1,
        )
