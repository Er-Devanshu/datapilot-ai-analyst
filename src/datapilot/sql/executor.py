from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.sql.validator import SQLValidator


@dataclass(frozen=True)
class SQLExecutionResult:
    """Result of a validated SQL execution."""

    sql: str
    dataframe: pd.DataFrame
    row_count: int


class SQLExecutor:
    """Execute SQL only after security validation."""

    def __init__(
        self,
        data_source: DuckDBDataSource,
        validator: SQLValidator | None = None,
    ) -> None:
        self.data_source = data_source
        self.validator = validator or SQLValidator()

    def execute(
        self,
        sql: str,
        allowed_tables: set[str] | None = None,
    ) -> SQLExecutionResult:
        """Validate and execute a read-only SQL query."""

        validation = self.validator.validate(
            sql=sql,
            allowed_tables=allowed_tables,
        )

        if not validation.valid:
            raise ValueError(
                "SQL validation failed: "
                + "; ".join(validation.errors)
            )

        dataframe = self.data_source.fetch_dataframe(
            validation.sql
        )

        return SQLExecutionResult(
            sql=validation.sql,
            dataframe=dataframe,
            row_count=len(dataframe),
        )