from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.data.schema import SchemaInspector
from datapilot.sql.policy import SQLSecurityPolicy
from datapilot.sql.validator import SQLValidator


@dataclass(frozen=True)
class SQLExecutionResult:
    """Result of a validated SQL execution."""

    sql: str
    dataframe: pd.DataFrame
    row_count: int


class SQLExecutor:
    """Execute SQL through the DataPilot security boundary."""

    def __init__(
        self,
        data_source: DuckDBDataSource,
        validator: SQLValidator | None = None,
        policy: SQLSecurityPolicy | None = None,
    ) -> None:
        self.data_source = data_source

        self.validator = (
            validator
            or SQLValidator()
        )

        self.policy = (
            policy
            or SQLSecurityPolicy.from_schema(
                SchemaInspector(data_source)
            )
        )

    def execute(
        self,
        sql: str,
    ) -> SQLExecutionResult:
        """Validate and execute a read-only SQL query."""

        validation = self.validator.validate(
            sql=sql,
            allowed_tables=set(
                self.policy.allowed_tables
            ),
            allowed_columns={
                table: set(columns)
                for table, columns
                in self.policy.allowed_columns.items()
            },
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