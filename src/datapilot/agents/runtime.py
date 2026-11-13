
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from datapilot.agents.graph import build_datapilot_graph
from datapilot.agents.state import DataPilotState
from datapilot.analytics.answer_validation import AnswerValidationResult
from datapilot.analytics.evidence import Evidence
from datapilot.analytics.executor import AnalyticsExecutionResult
from datapilot.analytics.planner import AnalyticsPlanningResult
from datapilot.analytics.planning import (
    AnalyticsPlan,
    AnalyticsPlanValidation,
)
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutor
from datapilot.visualization.models import VisualizationSpec
from datapilot.visualization.renderer import RenderedVisualization


@dataclass(frozen=True)
class AgentResponse:
    """Public response returned by the DataPilot agent."""

    question: str
    status: str
    sql: str | None
    result: pd.DataFrame | None
    row_count: int
    analysis: Any | None
    visualization: VisualizationSpec | None
    evidence: Evidence | None
    answer: str | None
    error: str | None = None
    rendered_visualization: RenderedVisualization | None = None

    # Structured analytics outputs.
    # These fields are optional to preserve compatibility with
    # existing AgentResponse construction and consumers.
    analytics_plan: AnalyticsPlan | None = None
    analytics_plan_validation: AnalyticsPlanValidation | None = None
    analytics_planning_result: AnalyticsPlanningResult | None = None
    analytics_execution: AnalyticsExecutionResult | None = None
    analytics_planning_error: str | None = None

    # Visualization provenance metadata.
    visualization_source: str | None = None
    visualization_request_id: str | None = None

    # Deterministic validation of the generated answer.
    answer_validation: AnswerValidationResult | None = None

    @classmethod
    def from_state(
        cls,
        state: DataPilotState,
    ) -> AgentResponse:
        """Create a public response from internal agent state."""

        return cls(
            question=state.get(
                "question",
                "",
            ),
            status=state.get(
                "status",
                "unknown",
            ),
            sql=state.get("sql"),
            result=state.get("result"),
            row_count=state.get(
                "row_count",
                0,
            ),
            analysis=state.get("analysis"),
            visualization=state.get("visualization"),
            evidence=state.get("evidence"),
            answer=state.get("answer"),
            error=state.get("error"),
            rendered_visualization=state.get(
                "rendered_visualization"
            ),
            analytics_plan=state.get("analytics_plan"),
            analytics_plan_validation=state.get(
                "analytics_plan_validation"
            ),
            analytics_planning_result=state.get(
                "analytics_planning_result"
            ),
            analytics_execution=state.get(
                "analytics_execution"
            ),
            analytics_planning_error=state.get(
                "analytics_planning_error"
            ),
            visualization_source=state.get(
                "visualization_source"
            ),
            visualization_request_id=state.get(
                "visualization_request_id"
            ),
            answer_validation=state.get(
                "answer_validation"
            ),
        )


class DataPilotAgent:
    """Public runtime for the DataPilot agent."""

    def __init__(
        self,
        llm: LocalLLM,
        inspector: SchemaInspector,
        executor: SQLExecutor,
        max_repair_attempts: int = 2,
        change_value_columns: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Initialize the DataPilot runtime."""

        if max_repair_attempts < 0:
            raise ValueError(
                "max_repair_attempts cannot be negative."
            )

        if change_value_columns is not None:
            if not isinstance(
                change_value_columns,
                (set, frozenset),
            ):
                raise ValueError(
                    "Change value columns must be a set or frozenset."
                )

            if not all(
                isinstance(column, str) and column.strip()
                for column in change_value_columns
            ):
                raise ValueError(
                    "Trusted change column names must be non-empty strings."
                )

            trusted_change_columns = frozenset(
                change_value_columns
            )
        else:
            trusted_change_columns = None

        self.llm = llm
        self.inspector = inspector
        self.executor = executor
        self.max_repair_attempts = max_repair_attempts
        self.change_value_columns = trusted_change_columns

        self.graph = build_datapilot_graph(
            llm=llm,
            inspector=inspector,
            executor=executor,
            max_repair_attempts=max_repair_attempts,
            change_value_columns=trusted_change_columns,
        )

    def ask(
        self,
        question: str,
    ) -> AgentResponse:
        """Execute the complete DataPilot pipeline."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        initial_state: DataPilotState = {
            "question": question,
            "status": "started",
        }

        try:
            result = self.graph.invoke(
                initial_state
            )
        except Exception as exc:
            return AgentResponse(
                question=question,
                status="failed",
                sql=None,
                result=None,
                row_count=0,
                analysis=None,
                visualization=None,
                evidence=None,
                answer=None,
                error=str(exc),
                rendered_visualization=None,
                answer_validation=None,
            )

        return AgentResponse.from_state(result)

    @staticmethod
    def _build_response(
        state: DataPilotState,
    ) -> AgentResponse:
        """Convert internal graph state into the public response."""

        return AgentResponse.from_state(state)
