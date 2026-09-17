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

    def __init__(
        self,
        max_sql_length: int = 10_000,
        max_result_rows: int = 10_000,
    ) -> None:
        self.max_sql_length = max_sql_length
        self.max_result_rows = max_result_rows

    def validate(
        self,
        sql: str,
        allowed_tables: set[str] | None = None,
        allowed_columns: dict[str, set[str]] | None = None,
    ) -> SQLValidationResult:
        """Validate SQL for safe read-only execution."""

        errors: list[str] = []

        sql = sql.strip()

        if not sql:
            errors.append(
                "SQL query cannot be empty."
            )

            return self._result(
                sql,
                errors,
            )

        if len(sql) > self.max_sql_length:
            errors.append(
                f"SQL query exceeds the maximum "
                f"length of {self.max_sql_length} characters."
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

            return self._result(
                sql,
                errors,
            )

        if len(statements) != 1:
            errors.append(
                "Exactly one SQL statement is required."
            )

            return self._result(
                sql,
                errors,
            )

        statement = statements[0]

        if isinstance(
            statement,
            self.BLOCKED_STATEMENTS,
        ):
            errors.append(
                "Write or schema-modifying SQL is not allowed."
            )

        if isinstance(statement, exp.Command):
            errors.append(
                "SQL command statements are not allowed."
            )

        if not isinstance(
            statement,
            exp.Query,
        ):
            errors.append(
                "Only read-only query statements are allowed."
            )

        if allowed_tables is not None:
            self._validate_tables(
                statement,
                allowed_tables,
                errors,
            )

        if allowed_columns is not None:
            self._validate_columns(
                statement,
                allowed_columns,
                errors,
            )

        self._validate_result_limit(
            statement,
            errors,
        )

        return self._result(
            sql,
            errors,
        )

    def _validate_tables(
        self,
        statement: exp.Expression,
        allowed_tables: set[str],
        errors: list[str],
    ) -> None:
        """Ensure referenced physical tables are authorized."""

        cte_names = {
            cte.alias_or_name
            for cte in statement.find_all(exp.CTE)
        }

        referenced_tables = {
            table.name
            for table in statement.find_all(exp.Table)
            if table.name not in cte_names
        }

        unauthorized = (
            referenced_tables - allowed_tables
        )

        if unauthorized:
            errors.append(
                "Unauthorized tables referenced: "
                f"{sorted(unauthorized)}"
            )

    def _validate_columns(
        self,
        statement: exp.Expression,
        allowed_columns: dict[str, set[str]],
        errors: list[str],
    ) -> None:
        """Ensure referenced physical columns are authorized."""

        all_columns = {
            column
            for columns in allowed_columns.values()
            for column in columns
        }

        table_aliases: dict[str, str] = {}

        for table in statement.find_all(exp.Table):
            table_name = table.name

            if table_name in allowed_columns:
                table_aliases[table_name] = table_name

                if table.alias:
                    table_aliases[table.alias] = table_name

        select_aliases = self._get_select_aliases(
            statement
        )

        invalid_columns: set[str] = set()

        for column in statement.find_all(exp.Column):
            column_name = column.name

            if column_name == "*":
                continue

            # A SELECT alias such as:
            #
            #   SUM(net_amount) AS revenue
            #
            # may legally be referenced later as:
            #
            #   ORDER BY revenue
            #
            if (
                not column.table
                and column_name in select_aliases
            ):
                continue

            qualifier = column.table

            if qualifier:
                table_name = table_aliases.get(
                    qualifier
                )

                if table_name is None:
                    invalid_columns.add(
                        f"{qualifier}.{column_name}"
                    )
                    continue

                if column_name not in allowed_columns[
                    table_name
                ]:
                    invalid_columns.add(
                        f"{table_name}.{column_name}"
                    )

            elif column_name not in all_columns:
                invalid_columns.add(
                    column_name
                )

        if invalid_columns:
            errors.append(
                "Unauthorized or unknown columns referenced: "
                f"{sorted(invalid_columns)}"
            )

    @staticmethod
    def _get_select_aliases(
        statement: exp.Expression,
    ) -> set[str]:
        """Return aliases defined by the SELECT projection."""

        aliases: set[str] = set()

        for select in statement.find_all(exp.Select):
            for expression in select.expressions:
                if isinstance(expression, exp.Alias):
                    alias = expression.alias

                    if alias:
                        aliases.add(alias)

        return aliases

    def _validate_result_limit(
        self,
        statement: exp.Expression,
        errors: list[str],
    ) -> None:
        """Prevent excessively large explicit result limits."""

        limit = statement.args.get("limit")

        if limit is None:
            return

        expression = limit.expression

        if isinstance(expression, exp.Literal):
            if expression.is_int:
                value = int(expression.this)

                if value > self.max_result_rows:
                    errors.append(
                        f"Query result limit {value:,} "
                        f"exceeds the maximum allowed "
                        f"limit of {self.max_result_rows:,} rows."
                    )

    @staticmethod
    def _result(
        sql: str,
        errors: list[str],
    ) -> SQLValidationResult:
        return SQLValidationResult(
            valid=not errors,
            sql=sql,
            errors=tuple(errors),
        )