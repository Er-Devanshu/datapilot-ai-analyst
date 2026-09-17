from __future__ import annotations

from dataclasses import dataclass

from datapilot.data.context import SchemaContextBuilder
from datapilot.data.schema import SchemaInspector
from datapilot.llm.local import LocalLLM
from datapilot.sql.executor import SQLExecutionResult, SQLExecutor
from datapilot.sql.generator import SQLGenerator
from datapilot.sql.repair import SQLRepairer
from datapilot.sql.validator import SQLValidationResult, SQLValidator


@dataclass(frozen=True)
class SQLPipelineResult:
    """Final result of the DataPilot SQL pipeline."""

    question: str
    sql: str
    execution: SQLExecutionResult
    repair_attempts: int


class SQLPipeline:
    """Generate, validate, repair, and execute SQL safely."""

    def __init__(
        self,
        llm: LocalLLM,
        inspector: SchemaInspector,
        executor: SQLExecutor,
        validator: SQLValidator | None = None,
        max_repair_attempts: int = 2,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError(
                "max_repair_attempts cannot be negative."
            )

        self.inspector = inspector
        self.executor = executor
        self.validator = (
            validator
            or SQLValidator()
        )

        self.schema_builder = (
            SchemaContextBuilder(inspector)
        )

        self.generator = SQLGenerator(llm)

        self.repairer = SQLRepairer(llm)

        self.max_repair_attempts = (
            max_repair_attempts
        )

    def run(
        self,
        question: str,
    ) -> SQLPipelineResult:
        """Run the complete SQL generation pipeline."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        schema_context = (
            self.schema_builder.build()
        )

        generation = self.generator.generate(
            question=question,
            schema_context=schema_context,
        )

        current_sql = generation.sql
        repair_attempts = 0

        while True:
            validation = self._validate(
                current_sql
            )

            if validation.valid:
                execution = self.executor.execute(
                    current_sql
                )

                return SQLPipelineResult(
                    question=question,
                    sql=current_sql,
                    execution=execution,
                    repair_attempts=repair_attempts,
                )

            if (
                repair_attempts
                >= self.max_repair_attempts
            ):
                raise RuntimeError(
                    "SQL validation failed after "
                    f"{repair_attempts} repair attempts: "
                    + "; ".join(validation.errors)
                )

            repair_attempts += 1

            repaired = self.repairer.repair(
                question=question,
                sql=current_sql,
                errors=validation.errors,
                schema_context=schema_context,
            )

            current_sql = repaired.repaired_sql

    def _validate(
        self,
        sql: str,
    ) -> SQLValidationResult:
        """Validate SQL using the executor's security policy."""

        return self.validator.validate(
            sql=sql,
            allowed_tables=set(
                self.executor.policy.allowed_tables
            ),
            allowed_columns={
                table: set(columns)
                for table, columns
                in self.executor.policy.allowed_columns.items()
            },
        )