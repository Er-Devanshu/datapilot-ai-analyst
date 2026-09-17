from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from datapilot.agents.nodes import (
    analysis_node,
    schema_node,
    sql_execution_node,
    sql_generation_node,
    sql_repair_node,
    sql_validation_node,
)
from datapilot.agents.state import DataPilotState
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutor


def build_datapilot_graph(
    llm: LocalLLM,
    inspector: SchemaInspector,
    executor: SQLExecutor,
    max_repair_attempts: int = 2,
):
    """Build the DataPilot LangGraph agent."""

    if max_repair_attempts < 0:
        raise ValueError(
            "max_repair_attempts cannot be negative."
        )

    graph = StateGraph(DataPilotState)

    graph.add_node(
        "schema",
        partial(
            schema_node,
            inspector=inspector,
        ),
    )

    graph.add_node(
        "generate_sql",
        partial(
            sql_generation_node,
            llm=llm,
        ),
    )

    graph.add_node(
        "validate_sql",
        partial(
            sql_validation_node,
            inspector=inspector,
        ),
    )

    graph.add_node(
        "repair_sql",
        partial(
            sql_repair_node,
            llm=llm,
        ),
    )

    graph.add_node(
        "execute_sql",
        partial(
            sql_execution_node,
            executor=executor,
        ),
    )

    graph.add_node(
        "analyze_result",
        analysis_node,
    )

    def route_after_validation(
        state: DataPilotState,
    ) -> str:
        """Choose execution, repair, or failure."""

        if state.get("sql_valid", False):
            return "execute"

        attempts = state.get(
            "repair_attempts",
            0,
        )

        if attempts < max_repair_attempts:
            return "repair"

        return "failure"

    def failure_node(
        state: DataPilotState,
    ) -> DataPilotState:
        """Mark the agent run as failed safely."""

        return {
            **state,
            "status": "failed",
            "error": (
                "SQL validation failed after "
                f"{max_repair_attempts} repair attempts."
            ),
        }

    graph.add_node(
        "failure",
        failure_node,
    )

    graph.add_edge(
        START,
        "schema",
    )

    graph.add_edge(
        "schema",
        "generate_sql",
    )

    graph.add_edge(
        "generate_sql",
        "validate_sql",
    )

    graph.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "execute": "execute_sql",
            "repair": "repair_sql",
            "failure": "failure",
        },
    )

    graph.add_edge(
        "repair_sql",
        "validate_sql",
    )

    graph.add_edge(
        "execute_sql",
        "analyze_result",
    )

    graph.add_edge(
        "analyze_result",
        END,
    )

    graph.add_edge(
        "failure",
        END,
    )

    return graph.compile()