from __future__ import annotations

from dataclasses import dataclass

from datapilot.data.schema import SchemaInspector


@dataclass(frozen=True)
class SQLSecurityPolicy:
    """Authorization policy derived from the database schema."""

    allowed_tables: frozenset[str]
    allowed_columns: dict[str, frozenset[str]]

    @classmethod
    def from_schema(
        cls,
        inspector: SchemaInspector,
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