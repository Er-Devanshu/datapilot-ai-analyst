from __future__ import annotations

import pandas as pd

from datapilot.agents.graph import build_datapilot_graph
from datapilot.agents.nodes import visualization_node
from datapilot.agents.state import DataPilotState
from datapilot.visualization.models import VisualizationSpec


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


def test_graph_contains_visualization_node() -> None:
    """Compiled graph should contain the visualization selection node."""

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
    assert "build_evidence" in graph_nodes