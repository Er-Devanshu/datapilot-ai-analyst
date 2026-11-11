
from __future__ import annotations

from typing import Any, TypedDict

from datapilot.analytics.executor import AnalyticsExecutionResult
from datapilot.analytics.planning import (
    AnalyticsPlan,
    AnalyticsPlanValidation,
)
from datapilot.analytics.planner import AnalyticsPlanningResult
from datapilot.visualization.models import VisualizationSpec


class DataPilotState(TypedDict, total=False):
    """State carried through the DataPilot agent graph."""

    question: str
    schema_context: str

    sql: str
    validation_errors: list[str]
    sql_valid: bool
    repair_attempts: int

    result: Any
    row_count: int

    # General deterministic result analysis.
    analysis: Any
    evidence: Any

    # Structured analytics planning and validation.
    analytics_plan: AnalyticsPlan
    analytics_plan_validation: AnalyticsPlanValidation
    analytics_planning_result: AnalyticsPlanningResult

    # Deterministic analytics execution.
    analytics_execution: AnalyticsExecutionResult

    visualization: VisualizationSpec | None

    # Rendered visualization metadata.
    #
    # We intentionally keep this as Any in the graph state rather than
    # storing a Plotly Figure directly in the TypedDict contract.
    # This keeps the state flexible for future output formats such as
    # HTML, JSON, PNG, or frontend-specific representations.
    rendered_visualization: Any

    answer: str

    status: str
    error: str
