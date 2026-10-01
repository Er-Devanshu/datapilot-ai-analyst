from __future__ import annotations

from dataclasses import dataclass

from datapilot.data.schema import SchemaInspector


@dataclass(frozen=True)
class SQLSecurityPolicy:
    """Authorization and execution policy for DataPilot SQL."""

    allowed_tables: frozenset[str]
    allowed_columns: dict[str, frozenset[str]]
    max_result_rows: int = 10_000

    def __post_init__(self) -> None:
        """Validate policy configuration."""

        if self.max_result_rows <= 0:
            raise ValueError(
                "max_result_rows must be greater than zero."
            )

    @classmethod
    def from_schema(
        cls,
        inspector: SchemaInspector,
        max_result_rows: int = 10_000,
    ) -> "SQLSecurityPolicy":
        """Build a security policy from the current database schema."""

        tables = inspector.inspect()

        allowed_tables = frozenset(
            table.name
            for table in tables
        )

        allowed_columns = {
            table.name: frozenset(
                column.name
                for column in table.columns
            )
            for table in tables
        }

        return cls(
            allowed_tables=allowed_tables,
            allowed_columns=allowed_columns,
            max_result_rows=max_result_rows,
        )

    def is_table_allowed(
        self,
        table_name: str,
    ) -> bool:
        """Return whether a table is authorized."""

        return table_name in self.allowed_tables

    def is_column_allowed(
        self,
        table_name: str,
        column_name: str,
    ) -> bool:
        """Return whether a column is authorized."""

        return column_name in self.allowed_columns.get(
            table_name,
            frozenset(),
        )