
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

    # Original SQL execution output. Never replaced by visualization data.
    result: Any
    row_count: int

    # General deterministic result analysis.
    analysis: Any
    evidence: Any

    # Structured analytics planning and validation.
    analytics_plan: AnalyticsPlan
    analytics_plan_validation: AnalyticsPlanValidation
    analytics_planning_result: AnalyticsPlanningResult
    analytics_planning_error: str | None

    # Deterministic analytics execution.
    analytics_execution: AnalyticsExecutionResult

    # Visualization selection and rendering.
    visualization: VisualizationSpec | None
    visualization_dataframe: Any
    visualization_source: str
    visualization_request_id: str | None
    rendered_visualization: Any

    answer: str

    status: str
    error: str
