
from __future__ import annotations

import pandas as pd
import pytest

from datapilot.agents.graph import build_datapilot_graph
from datapilot.agents.nodes import (
    evidence_node,
    visualization_node,
    visualization_rendering_node,
)
from datapilot.agents.state import DataPilotState
from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.visualization.models import VisualizationSpec
from datapilot.visualization.renderer import RenderedVisualization


def test_visualization_node_adds_specification() -> None:
    """Visualization node should add a deterministic chart specification."""

    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "West",
            ],
            "revenue": [
                100.0,
                200.0,
                150.0,
            ],
        }
    )

    state: DataPilotState = {
        "question": "What is revenue by region?",
        "result": dataframe,
    }

    result = visualization_node(state)

    visualization = result["visualization"]

    assert isinstance(
        visualization,
        VisualizationSpec,
    )

    assert visualization.chart_type == "bar"
    assert visualization.x_column == "region"
    assert visualization.y_column == "revenue"
    assert result["status"] == "visualization_selected"


def test_visualization_node_preserves_existing_state() -> None:
    """Visualization selection should preserve the existing agent state."""

    dataframe = pd.DataFrame(
        {
            "month": [
                "Jan",
                "Feb",
                "Mar",
            ],
            "revenue": [
                100.0,
                120.0,
                140.0,
            ],
        }
    )

    state: DataPilotState = {
        "question": "Show revenue trend.",
        "sql": (
            "SELECT month, revenue "
            "FROM monthly_revenue"
        ),
        "result": dataframe,
        "row_count": 3,
        "status": "analysis_complete",
    }

    result = visualization_node(state)

    assert result["question"] == state["question"]
    assert result["sql"] == state["sql"]
    assert result["row_count"] == 3
    assert result["status"] == "visualization_selected"
    assert result["visualization"].chart_type == "line"


def test_visualization_rendering_node_creates_plotly_figure() -> None:
    """Rendering node should create and store a rendered visualization."""

    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "West",
            ],
            "revenue": [
                100.0,
                200.0,
                150.0,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="revenue",
        title="Revenue by Region",
    )

    state: DataPilotState = {
        "question": "What is revenue by region?",
        "result": dataframe,
        "visualization": specification,
    }

    result = visualization_rendering_node(state)

    rendered = result["rendered_visualization"]

    assert isinstance(
        rendered,
        RenderedVisualization,
    )
    assert rendered.specification is specification
    assert len(rendered.figure.data) == 1
    assert rendered.figure.data[0].type == "bar"
    assert result["status"] == "visualization_rendered"


def test_visualization_rendering_node_rejects_missing_result() -> None:
    """Rendering requires an executed query result."""

    state: DataPilotState = {
        "question": "Show revenue.",
        "visualization": VisualizationSpec(
            chart_type="table",
        ),
    }

    with pytest.raises(
        ValueError,
        match="executed result",
    ):
        visualization_rendering_node(state)


def test_visualization_rendering_node_rejects_missing_specification() -> None:
    """Rendering requires a selected visualization specification."""

    state: DataPilotState = {
        "question": "Show revenue.",
        "result": pd.DataFrame(
            {
                "revenue": [100.0],
            }
        ),
    }

    with pytest.raises(
        ValueError,
        match="visualization specification",
    ):
        visualization_rendering_node(state)


def test_evidence_node_includes_selected_visualization() -> None:
    """Evidence should retain the visualization selected by the graph."""

    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "West",
            ],
            "revenue": [
                100.0,
                200.0,
                150.0,
            ],
        }
    )

    analysis = ResultAnalyzer().analyze(
        dataframe
    )

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="revenue",
        title="Revenue by Region",
    )

    state: DataPilotState = {
        "question": "What is revenue by region?",
        "result": dataframe,
        "analysis": analysis,
        "visualization": specification,
    }

    result = evidence_node(state)

    evidence = result["evidence"]

    assert evidence.visualization is specification
    assert evidence.visualization.chart_type == "bar"
    assert result["status"] == "evidence_ready"


def test_evidence_node_allows_missing_visualization() -> None:
    """Evidence generation remains valid without a visualization."""

    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    state: DataPilotState = {
        "question": "What is the revenue?",
        "result": dataframe,
        "analysis": ResultAnalyzer().analyze(dataframe),
    }

    result = evidence_node(state)

    assert result["evidence"].visualization is None


def test_graph_contains_visualization_pipeline_nodes() -> None:
    """Compiled graph should contain the complete visualization pipeline."""

    class StubLLM:
        pass

    class StubInspector:
        pass

    class StubExecutor:
        pass

    graph = build_datapilot_graph(
        llm=StubLLM(),
        inspector=StubInspector(),
        executor=StubExecutor(),
    )

    graph_nodes = graph.nodes

    assert "analyze_result" in graph_nodes
    assert "select_visualization" in graph_nodes
    assert "render_visualization" in graph_nodes
    assert "build_evidence" in graph_nodes
