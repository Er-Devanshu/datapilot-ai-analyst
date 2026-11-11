
from __future__ import annotations

import json

import pandas as pd
import pytest

from datapilot.agents.nodes import (
    evidence_node,
    visualization_node,
    visualization_rendering_node,
)
from datapilot.agents.runtime import AgentResponse
from datapilot.agents.state import DataPilotState
from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.executor import AnalyticsExecutor
from datapilot.analytics.planner import AnalyticsPlanner


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def build_segmentation_execution(
    dataframe: pd.DataFrame,
):
    response = {
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

    planning = AnalyticsPlanner(
        FakeLLM(json.dumps(response))
    ).plan(
        question="Show revenue by region.",
        dataframe=dataframe,
    )

    assert len(planning.validation.accepted_requests) == 1

    return AnalyticsExecutor().execute(
        dataframe=dataframe,
        validation=planning.validation,
    )


def test_visualization_node_selects_analytics_and_preserves_sql_result() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "North"],
            "revenue": [700.0, 200.0, 100.0],
        }
    )

    execution = build_segmentation_execution(dataframe)

    state: DataPilotState = {
        "question": "Show revenue by region.",
        "result": dataframe,
        "row_count": len(dataframe),
        "analytics_execution": execution,
    }

    result = visualization_node(state)

    assert result["visualization_source"] == "analytics"
    assert result["visualization_request_id"] == "revenue_by_region"
    assert result["visualization"].chart_type == "bar"
    assert result["visualization"].x_column == "segment"
    assert result["visualization"].y_column == "total_value"

    assert result["visualization_dataframe"] is not dataframe
    assert result["result"] is dataframe
    assert result["row_count"] == 3
    assert dataframe.columns.tolist() == ["region", "revenue"]


def test_visualization_node_falls_back_to_query_result() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [700.0, 200.0],
        }
    )

    result = visualization_node(
        {
            "question": "Show revenue.",
            "result": dataframe,
        }
    )

    assert result["visualization_source"] == "query_result"
    assert result["visualization_request_id"] is None
    assert result["visualization_dataframe"] is dataframe
    assert result["visualization"].x_column == "region"
    assert result["visualization"].y_column == "revenue"


def test_rendering_node_uses_visualization_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [700.0, 200.0],
        }
    )

    execution = build_segmentation_execution(dataframe)

    selected = visualization_node(
        {
            "question": "Show revenue by region.",
            "result": dataframe,
            "analytics_execution": execution,
        }
    )

    rendered = visualization_rendering_node(selected)

    figure = rendered["rendered_visualization"].figure

    assert rendered["visualization_source"] == "analytics"
    assert rendered["visualization_request_id"] == "revenue_by_region"
    assert list(figure.data[0].x) == ["West", "East"]
    assert list(figure.data[0].y) == [700.0, 200.0]


def test_evidence_node_records_visualization_provenance() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [700.0, 200.0],
        }
    )

    execution = build_segmentation_execution(dataframe)

    selected = visualization_node(
        {
            "question": "Show revenue by region.",
            "result": dataframe,
            "analysis": ResultAnalyzer().analyze(dataframe),
            "analytics_execution": execution,
        }
    )

    result = evidence_node(selected)
    evidence = result["evidence"]

    assert evidence.visualization_source == "analytics"
    assert evidence.visualization_request_id == "revenue_by_region"

    # Evidence summaries continue to describe the original SQL result.
    assert evidence.row_count == len(dataframe)
    assert evidence.result_columns == ("region", "revenue")

    prompt_text = evidence.to_prompt_text()

    assert "Visualization source: analytics" in prompt_text
    assert "Analytics request ID: revenue_by_region" in prompt_text


def test_evidence_node_records_query_result_fallback() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    selected = visualization_node(
        {
            "question": "Summarize revenue.",
            "result": dataframe,
            "analysis": ResultAnalyzer().analyze(dataframe),
        }
    )

    result = evidence_node(selected)
    evidence = result["evidence"]

    assert evidence.visualization_source == "query_result"
    assert evidence.visualization_request_id is None
    assert evidence.row_count == 2
    assert evidence.result_columns == ("revenue",)


def test_agent_response_exposes_visualization_provenance() -> None:
    response = AgentResponse.from_state(
        {
            "question": "Show revenue by region.",
            "status": "visualization_selected",
            "visualization_source": "analytics",
            "visualization_request_id": "revenue_by_region",
        }
    )

    assert response.visualization_source == "analytics"
    assert response.visualization_request_id == "revenue_by_region"


def test_agent_response_defaults_visualization_provenance_to_none() -> None:
    response = AgentResponse.from_state(
        {
            "question": "Summarize revenue.",
            "status": "visualization_selected",
        }
    )

    assert response.visualization_source is None
    assert response.visualization_request_id is None


def test_evidence_builder_rejects_analytics_source_without_request_id() -> None:
    dataframe = pd.DataFrame({"revenue": [100.0]})

    with pytest.raises(ValueError, match="requires a request ID"):
        EvidenceBuilder().build(
            dataframe=dataframe,
            visualization_source="analytics",
        )


def test_evidence_builder_rejects_unmatched_analytics_request_id() -> None:
    dataframe = pd.DataFrame({"revenue": [100.0]})

    with pytest.raises(
        ValueError,
        match="must match an analytics execution",
    ):
        EvidenceBuilder().build(
            dataframe=dataframe,
            analytics_executions=(),
            visualization_source="analytics",
            visualization_request_id="unknown_request",
        )


def test_evidence_builder_rejects_request_id_for_query_fallback() -> None:
    dataframe = pd.DataFrame({"revenue": [100.0]})

    with pytest.raises(
        ValueError,
        match="requires analytics as its source",
    ):
        EvidenceBuilder().build(
            dataframe=dataframe,
            visualization_source="query_result",
            visualization_request_id="unexpected_request",
        )
