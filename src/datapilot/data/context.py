from __future__ import annotations

from datapilot.data.schema import SchemaInspector, TableInfo


class SchemaContextBuilder:
    """Build compact schema context for LLM prompts."""

    def __init__(
        self,
        inspector: SchemaInspector,
    ) -> None:
        self.inspector = inspector

    def build(
        self,
        table_names: list[str] | None = None,
    ) -> str:
        """Build a textual representation of the database schema."""

        tables = self.inspector.inspect()

        if table_names is not None:
            requested = set(table_names)

            tables = tuple(
                table
                for table in tables
                if table.name in requested
            )

        if not tables:
            raise ValueError(
                "No tables available for schema context."
            )

        sections = [
            self._format_table(table)
            for table in tables
        ]

        return "\n\n".join(sections)

    @staticmethod
    def _format_table(
        table: TableInfo,
    ) -> str:
        """Format one table for an LLM prompt."""

        lines = [
            f"TABLE: {table.name}",
            "COLUMNS:",
        ]

        for column in table.columns:
            nullable = "NULLABLE" if column.nullable else "NOT NULL"

            lines.append(
                f"  - {column.name}: "
                f"{column.data_type} "
                f"({nullable})"
            )

        return "\n".join(lines)