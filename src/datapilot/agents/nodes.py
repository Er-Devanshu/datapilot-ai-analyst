from __future__ import annotations

from datapilot.agents.state import DataPilotState
from datapilot.data.context import SchemaContextBuilder
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutor
from datapilot.sql.generator import SQLGenerator
from datapilot.sql.repair import SQLRepairer
from datapilot.sql.validator import SQLValidator


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

    question = state.get("question", "").strip()
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

    generator = SQLGenerator(llm)

    result = generator.generate(
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

    sql = state.get("sql", "").strip()

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

    validator = SQLValidator()

    result = validator.validate(
        sql=sql,
        allowed_tables=policy_tables,
        allowed_columns=policy_columns,
    )

    return {
        **state,
        "sql_valid": result.valid,
        "validation_errors": list(
            result.errors
        ),
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

    question = state.get("question", "").strip()
    sql = state.get("sql", "").strip()
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

    repairer = SQLRepairer(llm)

    result = repairer.repair(
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

    sql = state.get("sql", "").strip()
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
            "Cannot execute SQL that has not "
            "passed validation."
        )

    result = executor.execute(sql)

    return {
        **state,
        "result": result.dataframe,
        "row_count": result.row_count,
        "status": "sql_executed",
    }