from __future__ import annotations

from typing import Any, TypedDict

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

    analysis: Any
    evidence: Any

    visualization: VisualizationSpec | None

    answer: str
    status: str
    error: str