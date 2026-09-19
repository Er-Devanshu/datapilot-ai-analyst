from __future__ import annotations

from typing import Any, TypedDict


class DataPilotState(TypedDict, total=False):
    """State carried through the DataPilot agent graph."""

    # User input
    question: str

    # Database understanding
    schema_context: str

    # SQL generation
    sql: str

    # SQL validation
    validation_errors: list[str]
    sql_valid: bool

    # SQL repair
    repair_attempts: int

    # Query execution
    result: Any
    row_count: int

    # Analytical reasoning
    analysis: Any
    evidence: Any

    # Final answer
    answer: str

    # Agent lifecycle
    status: str
    error: str