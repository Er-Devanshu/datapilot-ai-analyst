from __future__ import annotations

import pytest

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.sql.executor import SQLExecutor
from datapilot.sql.policy import SQLSecurityPolicy


def build_test_database() -> DuckDBDataSource:
    """Create a deterministic database for executor tests."""

    data_source = DuckDBDataSource(":memory:")

    data_source.execute(
        """
        CREATE TABLE sales (
            region VARCHAR,
            revenue DOUBLE
        )
        """
    )

    data_source.execute(
        """
        INSERT INTO sales
        VALUES
            ('West', 700),
            ('East', 200),
            ('North', 100)
        """
    )

    return data_source


def build_executor(
    database: DuckDBDataSource,
    max_result_rows: int = 10_000,
) -> SQLExecutor:
    """Create an executor with an explicit security policy."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {
                    "region",
                    "revenue",
                }
            )
        },
        max_result_rows=max_result_rows,
    )

    return SQLExecutor(
        data_source=database,
        policy=policy,
    )


def test_executor_executes_valid_sql() -> None:
    """A valid authorized query should execute successfully."""

    database = build_test_database()
    executor = build_executor(database)

    result = executor.execute(
        """
        SELECT
            region,
            revenue
        FROM sales
        ORDER BY revenue DESC
        """
    )

    assert result.row_count == 3
    assert list(result.dataframe["region"]) == [
        "West",
        "East",
        "North",
    ]

    database.close()


def test_executor_returns_validated_sql() -> None:
    """The execution result should expose the validated SQL."""

    database = build_test_database()
    executor = build_executor(database)

    result = executor.execute(
        "SELECT region, revenue FROM sales"
    )

    assert result.sql.strip().upper().startswith(
        "SELECT"
    )

    assert "sales" in result.sql.lower()

    database.close()


def test_executor_rejects_unauthorized_table() -> None:
    """Queries against unauthorized tables must fail."""

    database = build_test_database()
    executor = build_executor(database)

    with pytest.raises(
        ValueError,
        match="SQL validation failed",
    ):
        executor.execute(
            "SELECT * FROM secret_table"
        )

    database.close()


def test_executor_rejects_unauthorized_column() -> None:
    """Queries against unauthorized columns must fail."""

    database = build_test_database()
    executor = build_executor(database)

    with pytest.raises(
        ValueError,
        match="SQL validation failed",
    ):
        executor.execute(
            "SELECT customer_id FROM sales"
        )

    database.close()


def test_executor_rejects_non_select_sql() -> None:
    """Write operations must not pass the SQL security boundary."""

    database = build_test_database()
    executor = build_executor(database)

    with pytest.raises(
        ValueError,
        match="SQL validation failed",
    ):
        executor.execute(
            """
            DELETE FROM sales
            """
        )

    database.close()


def test_executor_reports_row_count() -> None:
    """row_count should equal the number of returned rows."""

    database = build_test_database()
    executor = build_executor(database)

    result = executor.execute(
        "SELECT region FROM sales"
    )

    assert result.row_count == len(
        result.dataframe
    )

    assert result.row_count == 3

    database.close()


def test_executor_enforces_max_result_rows() -> None:
    """The executor must reject results exceeding the policy limit."""

    database = build_test_database()
    executor = build_executor(
        database,
        max_result_rows=2,
    )

    with pytest.raises(
        ValueError,
        match="maximum allowed result rows",
    ):
        executor.execute(
            """
            SELECT
                region,
                revenue
            FROM sales
            ORDER BY region
            """
        )

    database.close()


def test_executor_allows_result_at_exact_limit() -> None:
    """A result exactly at the configured limit should be accepted."""

    database = build_test_database()
    executor = build_executor(
        database,
        max_result_rows=3,
    )

    result = executor.execute(
        """
        SELECT
            region,
            revenue
        FROM sales
        ORDER BY region
        """
    )

    assert result.row_count == 3

    database.close()


def test_executor_allows_single_row_result_with_limit_one() -> None:
    """A single-row result should be accepted when the limit is one."""

    database = build_test_database()
    executor = build_executor(
        database,
        max_result_rows=1,
    )

    result = executor.execute(
        """
        SELECT
            region,
            revenue
        FROM sales
        WHERE region = 'West'
        """
    )

    assert result.row_count == 1

    database.close()


def test_executor_preserves_empty_result() -> None:
    """An empty valid result should remain an empty DataFrame."""

    database = build_test_database()
    executor = build_executor(database)

    result = executor.execute(
        """
        SELECT
            region,
            revenue
        FROM sales
        WHERE region = 'DoesNotExist'
        """
    )

    assert result.row_count == 0
    assert result.dataframe.empty

    database.close()