from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from datapilot.agents.graph import build_datapilot_graph
from datapilot.agents.state import DataPilotState
from datapilot.analytics.evidence import Evidence
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutor
from datapilot.visualization.models import VisualizationSpec


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
            sql=state.get(
                "sql"
            ),
            result=state.get(
                "result"
            ),
            row_count=state.get(
                "row_count",
                0,
            ),
            analysis=state.get(
                "analysis"
            ),
            visualization=state.get(
                "visualization"
            ),
            evidence=state.get(
                "evidence"
            ),
            answer=state.get(
                "answer"
            ),
            error=state.get(
                "error"
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
    ) -> None:
        """Initialize the DataPilot runtime."""

        if max_repair_attempts < 0:
            raise ValueError(
                "max_repair_attempts cannot be negative."
            )

        self.llm = llm
        self.inspector = inspector
        self.executor = executor
        self.max_repair_attempts = max_repair_attempts

        self.graph = build_datapilot_graph(
            llm=llm,
            inspector=inspector,
            executor=executor,
            max_repair_attempts=max_repair_attempts,
        )

    def ask(
        self,
        question: str,
    ) -> AgentResponse:
        """Execute the complete DataPilot pipeline."""

        if not question.strip():
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
            )

        return AgentResponse.from_state(
            result
        )

    @staticmethod
    def _build_response(
        state: DataPilotState,
    ) -> AgentResponse:
        """Convert internal graph state into the public response."""

        return AgentResponse.from_state(
            state
        )