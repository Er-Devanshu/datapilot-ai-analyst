from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class SQLValidationResult:
    """Result of SQL security and safety validation."""

    valid: bool
    sql: str
    errors: tuple[str, ...]


class SQLValidator:
    """Validate LLM-generated SQL before database execution."""

    BLOCKED_STATEMENTS = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.TruncateTable,
        exp.Merge,
    )

    def validate(
        self,
        sql: str,
        allowed_tables: set[str] | None = None,
    ) -> SQLValidationResult:
        """Validate a SQL statement for safe execution."""

        errors: list[str] = []

        if not sql.strip():
            errors.append("SQL query cannot be empty.")

            return SQLValidationResult(
                valid=False,
                sql=sql,
                errors=tuple(errors),
            )

        try:
            statements = sqlglot.parse(
                sql,
                dialect="duckdb",
            )
        except Exception as exc:
            errors.append(
                f"SQL parsing failed: {exc}"
            )

            return SQLValidationResult(
                valid=False,
                sql=sql,
                errors=tuple(errors),
            )

        if len(statements) != 1:
            errors.append(
                "Exactly one SQL statement is required."
            )

        statement = statements[0]

        if isinstance(
            statement,
            self.BLOCKED_STATEMENTS,
        ):
            errors.append(
                "Write or schema-modifying SQL is not allowed."
            )

        if not isinstance(
            statement,
            exp.Query,
        ):
            errors.append(
                "Only read-only query statements are allowed."
            )

        if allowed_tables is not None:
            referenced_tables = {
                table.name
                for table in statement.find_all(exp.Table)
            }

            unauthorized = (
                referenced_tables - allowed_tables
            )

            if unauthorized:
                errors.append(
                    "Unauthorized tables referenced: "
                    f"{sorted(unauthorized)}"
                )

        return SQLValidationResult(
            valid=not errors,
            sql=sql,
            errors=tuple(errors),
        )