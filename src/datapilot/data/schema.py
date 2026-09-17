from __future__ import annotations

from dataclasses import dataclass

from datapilot.data.duckdb import DuckDBDataSource


@dataclass(frozen=True)
class ColumnInfo:
    """Metadata describing a database column."""

    name: str
    data_type: str
    nullable: bool


@dataclass(frozen=True)
class TableInfo:
    """Metadata describing a database table."""

    name: str
    columns: tuple[ColumnInfo, ...]


class SchemaInspector:
    """Inspect the schema available in a DuckDB database."""

    def __init__(self, data_source: DuckDBDataSource) -> None:
        self.data_source = data_source

    def list_tables(self) -> list[str]:
        """Return all user-visible tables."""

        result = self.data_source.fetch_dataframe(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )

        return result["table_name"].tolist()

    def describe_table(self, table_name: str) -> TableInfo:
        """Return column metadata for a table."""

        if not table_name.strip():
            raise ValueError("Table name cannot be empty.")

        if table_name not in self.list_tables():
            raise ValueError(f"Table does not exist: {table_name}")

        escaped_table_name = table_name.replace("'", "''")

        result = self.data_source.fetch_dataframe(
            f"""
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = '{escaped_table_name}'
            ORDER BY ordinal_position
            """
        )

        columns = tuple(
            ColumnInfo(
                name=row["column_name"],
                data_type=row["data_type"],
                nullable=row["is_nullable"] == "YES",
            )
            for _, row in result.iterrows()
        )

        return TableInfo(
            name=table_name,
            columns=columns,
        )

    def inspect(self) -> tuple[TableInfo, ...]:
        """Return metadata for all available tables."""

        return tuple(
            self.describe_table(table_name)
            for table_name in self.list_tables()
        )