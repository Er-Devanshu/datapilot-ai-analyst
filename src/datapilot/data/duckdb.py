from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


class DuckDBDataSource:
    """DuckDB data access layer for DataPilot."""

    def __init__(
        self,
        database_path: str | Path = ":memory:",
        read_only: bool = False,
    ) -> None:
        self.database_path = str(database_path)
        self.read_only = read_only
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        """Open the DuckDB connection."""

        if self._connection is not None:
            return

        if self.database_path != ":memory:" and not self.read_only:
            Path(self.database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = duckdb.connect(
            database=self.database_path,
            read_only=self.read_only,
        )

    def close(self) -> None:
        """Close the DuckDB connection."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def execute(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query and return the DuckDB relation."""

        if not query.strip():
            raise ValueError("SQL query cannot be empty.")

        self.connect()

        if self._connection is None:
            raise RuntimeError("DuckDB connection could not be established.")

        return self._connection.sql(query)

    def fetch_dataframe(self, query: str) -> Any:
        """Execute a SQL query and return the result as a DataFrame."""

        return self.execute(query).df()

    def __enter__(self) -> "DuckDBDataSource":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()