
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from datapilot.agents.runtime import (
    AgentResponse,
    DataPilotAgent,
)
from datapilot.data.schema import (
    ColumnInfo,
    TableInfo,
)
from datapilot.visualization.renderer import RenderedVisualization


@dataclass
class FakeLLM:
    """Deterministic LLM stub for runtime integration tests."""

    def generate(self, prompt: str) -> str:
        """Return deterministic SQL or answer content."""

        if "ANSWER" in prompt.upper():
            return (
                "West generated the highest revenue with "
                "₹700."
            )

        return (
            "SELECT region, SUM(revenue) AS revenue "
            "FROM sales "
            "GROUP BY region"
        )


class FakeSchemaInspector:
    """Deterministic schema inspector for integration tests."""

    def inspect(self) -> tuple[TableInfo, ...]:
        """Return a minimal sales schema."""

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
    """Deterministic SQL executor for integration tests."""

    def execute(self, sql: str):
        """Return a deterministic query result."""

        assert "SELECT" in sql.upper()
        assert "sales" in sql.lower()

        dataframe = pd.DataFrame(
            {
                "region": [
                    "West",
                    "East",
                    "North",
                ],
                "revenue": [
                    700.0,
                    200.0,
                    100.0,
                ],
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


def build_agent() -> DataPilotAgent:
    """Build a deterministic agent for integration tests."""

    return DataPilotAgent(
        llm=FakeLLM(),
        inspector=FakeSchemaInspector(),
        executor=FakeSQLExecutor(),
    )


def test_runtime_executes_complete_agent_pipeline() -> None:
    """The public runtime should execute the complete agent pipeline."""

    agent = build_agent()

    response = agent.ask(
        "What is revenue by region?"
    )

    assert isinstance(
        response,
        AgentResponse,
    )

    assert response.question == (
        "What is revenue by region?"
    )

    assert response.status == "answer_composed"
    assert response.sql is not None
    assert response.result is not None
    assert response.row_count == 3
    assert response.analysis is not None
    assert response.visualization is not None
    assert response.evidence is not None
    assert response.answer is not None
    assert response.rendered_visualization is not None


def test_runtime_selects_bar_visualization_for_dimension_measure() -> None:
    """Categorical dimension + numeric measure should produce a bar chart."""

    response = build_agent().ask(
        "What is revenue by region?"
    )

    assert response.visualization is not None
    assert response.visualization.chart_type == "bar"
    assert response.visualization.x_column == "region"
    assert response.visualization.y_column == "revenue"


def test_runtime_builds_grounded_evidence() -> None:
    """The complete runtime should produce deterministic evidence."""

    response = build_agent().ask(
        "What is revenue by region?"
    )

    assert response.evidence is not None
    assert response.evidence.row_count == 3

    assert response.evidence.result_columns == (
        "region",
        "revenue",
    )

    assert response.evidence.currency == "INR"

    assert (
        response.evidence.numeric_summaries[0].total
        == 1000
    )

    assert (
        response.evidence.category_summaries[0].top_category
        == "West"
    )

    assert (
        response.evidence.category_summaries[0].top_value
        == 700
    )


def test_runtime_exposes_rendered_plotly_figure() -> None:
    """The public response should expose the rendered Plotly figure."""

    response = build_agent().ask(
        "What is revenue by region?"
    )

    rendered = response.rendered_visualization

    assert isinstance(
        rendered,
        RenderedVisualization,
    )

    assert isinstance(
        rendered.figure,
        go.Figure,
    )

    assert rendered.specification is response.visualization

    assert rendered.specification.chart_type == "bar"
    assert rendered.specification.x_column == "region"
    assert rendered.specification.y_column == "revenue"

    # Plotly Express creates one trace per color category.
    assert len(rendered.figure.data) == 3

    actual_values = {
        trace.name: (
            list(trace.x),
            list(trace.y),
        )
        for trace in rendered.figure.data
    }

    assert actual_values == {
        "West": (["West"], [700.0]),
        "East": (["East"], [200.0]),
        "North": (["North"], [100.0]),
    }


def test_runtime_evidence_uses_same_visualization_specification() -> None:
    """Evidence and rendered output must use the same chart specification."""

    response = build_agent().ask(
        "What is revenue by region?"
    )

    assert response.evidence is not None
    assert response.visualization is not None
    assert response.rendered_visualization is not None

    assert (
        response.evidence.visualization
        is response.visualization
    )

    assert (
        response.rendered_visualization.specification
        is response.visualization
    )


def test_runtime_preserves_question() -> None:
    """The public runtime should preserve the user's question."""

    question = "Show me revenue by region."

    response = build_agent().ask(question)

    assert response.question == question


def test_runtime_returns_final_answer() -> None:
    """The complete pipeline should produce a final answer."""

    response = build_agent().ask(
        "What is revenue by region?"
    )

    assert response.answer is not None
    assert isinstance(response.answer, str)
