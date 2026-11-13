
from __future__ import annotations

import pandas as pd

from datapilot.agents.state import DataPilotState
from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.answer import AnswerComposer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.executor import AnalyticsExecutor
from datapilot.analytics.planner import (
    AnalyticsPlanGenerationError,
    AnalyticsPlanner,
    AnalyticsPlanningResult,
)
from datapilot.analytics.planning import (
    AnalyticsPlan,
    AnalyticsPlanValidation,
    PlanValidationIssue,
)
from datapilot.data.context import SchemaContextBuilder
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutor
from datapilot.sql.generator import SQLGenerator
from datapilot.sql.repair import SQLRepairer
from datapilot.sql.validator import SQLValidator
from datapilot.visualization.analytics_selection import (
    AnalyticsAwareVisualizationSelector,
)
from datapilot.visualization.renderer import VisualizationRenderer


def schema_node(
    state: DataPilotState,
    inspector: SchemaInspector,
) -> DataPilotState:
    """Populate the agent state with database schema context."""

    question = state.get("question", "").strip()

    if not question:
        raise ValueError(
            "Agent state must contain a question."
        )

    schema_context = SchemaContextBuilder(
        inspector
    ).build()

    return {
        **state,
        "schema_context": schema_context,
        "status": "schema_ready",
    }


def sql_generation_node(
    state: DataPilotState,
    llm: LocalLLM,
) -> DataPilotState:
    """Generate SQL from the current agent state."""

    question = state.get(
        "question",
        "",
    ).strip()

    schema_context = state.get(
        "schema_context",
        "",
    ).strip()

    if not question:
        raise ValueError(
            "Agent state must contain a question."
        )

    if not schema_context:
        raise ValueError(
            "Agent state must contain schema context."
        )

    result = SQLGenerator(llm).generate(
        question=question,
        schema_context=schema_context,
    )

    return {
        **state,
        "sql": result.sql,
        "status": "sql_generated",
    }


def sql_validation_node(
    state: DataPilotState,
    inspector: SchemaInspector,
) -> DataPilotState:
    """Validate generated SQL against the database schema."""

    sql = state.get(
        "sql",
        "",
    ).strip()

    if not sql:
        raise ValueError(
            "Agent state must contain SQL."
        )

    schema = inspector.inspect()

    policy_tables = {
        table.name
        for table in schema
    }

    policy_columns = {
        table.name: {
            column.name
            for column in table.columns
        }
        for table in schema
    }

    result = SQLValidator().validate(
        sql=sql,
        allowed_tables=policy_tables,
        allowed_columns=policy_columns,
    )

    return {
        **state,
        "sql_valid": result.valid,
        "validation_errors": list(result.errors),
        "status": (
            "sql_validated"
            if result.valid
            else "sql_invalid"
        ),
    }


def sql_repair_node(
    state: DataPilotState,
    llm: LocalLLM,
) -> DataPilotState:
    """Repair invalid SQL using validation feedback."""

    question = state.get(
        "question",
        "",
    ).strip()

    sql = state.get(
        "sql",
        "",
    ).strip()

    schema_context = state.get(
        "schema_context",
        "",
    ).strip()

    validation_errors = state.get(
        "validation_errors",
        [],
    )

    repair_attempts = state.get(
        "repair_attempts",
        0,
    )

    if not question:
        raise ValueError(
            "Agent state must contain a question."
        )

    if not sql:
        raise ValueError(
            "Agent state must contain SQL."
        )

    if not schema_context:
        raise ValueError(
            "Agent state must contain schema context."
        )

    if not validation_errors:
        raise ValueError(
            "Agent state must contain validation errors."
        )

    result = SQLRepairer(llm).repair(
        question=question,
        sql=sql,
        errors=tuple(validation_errors),
        schema_context=schema_context,
    )

    return {
        **state,
        "sql": result.repaired_sql,
        "repair_attempts": repair_attempts + 1,
        "status": "sql_repaired",
    }


def sql_execution_node(
    state: DataPilotState,
    executor: SQLExecutor,
) -> DataPilotState:
    """Execute validated SQL through the security boundary."""

    sql = state.get(
        "sql",
        "",
    ).strip()

    sql_valid = state.get(
        "sql_valid",
        False,
    )

    if not sql:
        raise ValueError(
            "Agent state must contain SQL."
        )

    if not sql_valid:
        raise ValueError(
            "Cannot execute SQL that has not passed validation."
        )

    result = executor.execute(sql)

    return {
        **state,
        "result": result.dataframe,
        "row_count": result.row_count,
        "status": "sql_executed",
    }


def analysis_node(
    state: DataPilotState,
) -> DataPilotState:
    """Analyze the executed query result deterministically."""

    result = state.get("result")

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    analysis = ResultAnalyzer().analyze(result)

    return {
        **state,
        "analysis": analysis,
        "status": "analysis_complete",
    }


def analytics_planning_node(
    state: DataPilotState,
    llm: LocalLLM,
    change_value_columns: set[str] | frozenset[str] | None = None,
) -> DataPilotState:
    """Generate an analytics plan without blocking the core agent."""

    question = state.get(
        "question",
        "",
    ).strip()

    result = state.get("result")

    if not question:
        raise ValueError(
            "Agent state must contain a question."
        )

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "Executed result must be a pandas DataFrame."
        )

    try:
        planning_result = AnalyticsPlanner(
            llm=llm,
        ).plan(
            question=question,
            dataframe=result,
            change_value_columns=change_value_columns,
        )

    except AnalyticsPlanGenerationError as exc:
        plan = AnalyticsPlan(
            question=question,
            requests=(),
        )

        validation = AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=(),
            fatal_errors=(
                PlanValidationIssue(
                    code="PLAN_GENERATION_FAILED",
                    message=str(exc),
                ),
            ),
        )

        planning_result = AnalyticsPlanningResult(
            plan=plan,
            validation=validation,
            prompt="",
            model_name="unknown",
        )

        return {
            **state,
            "analytics_plan": plan,
            "analytics_plan_validation": validation,
            "analytics_planning_result": planning_result,
            "analytics_planning_error": str(exc),
            "status": "analytics_plan_failed",
        }

    return {
        **state,
        "analytics_plan": planning_result.plan,
        "analytics_plan_validation": planning_result.validation,
        "analytics_planning_result": planning_result,
        "analytics_planning_error": None,
        "status": "analytics_plan_ready",
    }


def analytics_execution_node(
    state: DataPilotState,
    executor: AnalyticsExecutor | None = None,
) -> DataPilotState:
    """Execute the accepted deterministic analytics requests."""

    result = state.get("result")

    validation = state.get(
        "analytics_plan_validation"
    )

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "Executed result must be a pandas DataFrame."
        )

    if validation is None:
        raise ValueError(
            "Agent state must contain analytics plan validation."
        )

    analytics_executor = (
        executor
        if executor is not None
        else AnalyticsExecutor()
    )

    execution_result = analytics_executor.execute(
        dataframe=result,
        validation=validation,
    )

    return {
        **state,
        "analytics_execution": execution_result,
        "status": "analytics_executed",
    }


def visualization_node(
    state: DataPilotState,
) -> DataPilotState:
    """Select a visualization from analytics output or query results."""

    result = state.get("result")

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "Executed result must be a pandas DataFrame."
        )

    analytics_execution = state.get("analytics_execution")

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe=result,
        analytics_execution=analytics_execution,
    )

    return {
        **state,
        "visualization": selection.specification,
        "visualization_dataframe": selection.dataframe,
        "visualization_source": selection.source,
        "visualization_request_id": selection.request_id,
        "status": "visualization_selected",
    }


def visualization_rendering_node(
    state: DataPilotState,
) -> DataPilotState:
    """Render the selected visualization using its selected DataFrame."""

    result = state.get("result")
    visualization_dataframe = state.get(
        "visualization_dataframe"
    )
    visualization = state.get("visualization")

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    if visualization is None:
        raise ValueError(
            "Agent state must contain a visualization specification."
        )

    if visualization_dataframe is None:
        visualization_dataframe = result

    if not isinstance(visualization_dataframe, pd.DataFrame):
        raise TypeError(
            "Visualization dataframe must be a pandas DataFrame."
        )

    rendered_visualization = VisualizationRenderer().render(
        dataframe=visualization_dataframe,
        specification=visualization,
    )

    return {
        **state,
        "rendered_visualization": rendered_visualization,
        "status": "visualization_rendered",
    }


def evidence_node(
    state: DataPilotState,
) -> DataPilotState:
    """Build grounded evidence from all deterministic analyses."""

    result = state.get("result")
    analysis = state.get("analysis")
    visualization = state.get("visualization")
    analytics_execution = state.get("analytics_execution")
    visualization_source = state.get("visualization_source")
    visualization_request_id = state.get(
        "visualization_request_id"
    )

    if result is None:
        raise ValueError(
            "Agent state must contain an executed result."
        )

    if analysis is None:
        raise ValueError(
            "Agent state must contain analysis."
        )

    evidence = EvidenceBuilder().build(
        dataframe=result,
        analysis=analysis,
        visualization=visualization,
        analytics_execution_result=analytics_execution,
        visualization_source=visualization_source,
        visualization_request_id=visualization_request_id,
    )

    return {
        **state,
        "evidence": evidence,
        "status": "evidence_ready",
    }


def answer_composer_node(
    state: DataPilotState,
    llm: LocalLLM,
) -> DataPilotState:
    """Compose and validate a business answer from grounded evidence."""

    question = state.get(
        "question",
        "",
    ).strip()

    evidence = state.get("evidence")

    if not question:
        raise ValueError(
            "Agent state must contain a question."
        )

    if evidence is None:
        raise ValueError(
            "Agent state must contain evidence."
        )

    result = AnswerComposer(llm).compose(
        question=question,
        evidence=evidence,
    )

    return {
        **state,
        "answer": result.answer,
        "answer_validation": result.validation,
        "status": "answer_composed",
    }
