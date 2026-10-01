from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datapilot.data.duckdb import DuckDBDataSource


def test_in_memory_datasource_connects() -> None:
    """An in-memory datasource should establish a DuckDB connection."""

    data_source = DuckDBDataSource()

    data_source.connect()

    assert data_source._connection is not None

    data_source.close()


def test_connect_is_idempotent() -> None:
    """Calling connect repeatedly should reuse the existing connection."""

    data_source = DuckDBDataSource()

    data_source.connect()
    first_connection = data_source._connection

    data_source.connect()
    second_connection = data_source._connection

    assert first_connection is not None
    assert second_connection is first_connection

    data_source.close()


def test_execute_returns_duckdb_relation() -> None:
    """Execute should return a DuckDB relation for valid SQL."""

    data_source = DuckDBDataSource()

    relation = data_source.execute(
        "SELECT 1 AS value"
    )

    assert relation is not None
    assert relation.fetchall() == [(1,)]

    data_source.close()


def test_execute_rejects_empty_sql() -> None:
    """Execute should reject blank SQL statements."""

    data_source = DuckDBDataSource()

    with pytest.raises(
        ValueError,
        match="SQL query cannot be empty",
    ):
        data_source.execute("   ")

    data_source.close()


def test_fetch_dataframe_returns_dataframe() -> None:
    """fetch_dataframe should return a pandas DataFrame."""

    data_source = DuckDBDataSource()

    dataframe = data_source.fetch_dataframe(
        """
        SELECT
            'West' AS region,
            700 AS revenue
        """
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert list(dataframe.columns) == [
        "region",
        "revenue",
    ]

    assert len(dataframe) == 1

    assert dataframe.iloc[0]["region"] == "West"
    assert dataframe.iloc[0]["revenue"] == 700

    data_source.close()


def test_close_releases_connection() -> None:
    """close should release the active DuckDB connection."""

    data_source = DuckDBDataSource()

    data_source.connect()

    assert data_source._connection is not None

    data_source.close()

    assert data_source._connection is None


def test_close_is_idempotent() -> None:
    """Calling close repeatedly should be safe."""

    data_source = DuckDBDataSource()

    data_source.connect()
    data_source.close()
    data_source.close()

    assert data_source._connection is None


def test_context_manager_connects_and_closes() -> None:
    """The context manager should manage the connection lifecycle."""

    data_source = DuckDBDataSource()

    with data_source as active_source:
        assert active_source is data_source
        assert data_source._connection is not None

        dataframe = data_source.fetch_dataframe(
            "SELECT 42 AS answer"
        )

        assert dataframe.iloc[0]["answer"] == 42

    assert data_source._connection is None


def test_file_backed_database_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """A writable file-backed database should create missing directories."""

    database_path = (
        tmp_path
        / "nested"
        / "database"
        / "datapilot.duckdb"
    )

    data_source = DuckDBDataSource(
        database_path=database_path,
    )

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
            ('East', 200)
        """
    )

    dataframe = data_source.fetch_dataframe(
        """
        SELECT
            region,
            revenue
        FROM sales
        ORDER BY revenue DESC
        """
    )

    assert database_path.exists()
    assert list(dataframe["region"]) == [
        "West",
        "East",
    ]
    assert list(dataframe["revenue"]) == [
        700.0,
        200.0,
    ]

    data_source.close()


def test_file_backed_database_persists_data(
    tmp_path: Path,
) -> None:
    """Data in a file-backed database should survive reconnects."""

    database_path = (
        tmp_path
        / "datapilot.duckdb"
    )

    first_source = DuckDBDataSource(
        database_path=database_path,
    )

    first_source.execute(
        """
        CREATE TABLE sales (
            region VARCHAR,
            revenue DOUBLE
        )
        """
    )

    first_source.execute(
        """
        INSERT INTO sales
        VALUES ('West', 700)
        """
    )

    first_source.close()

    second_source = DuckDBDataSource(
        database_path=database_path,
    )

    dataframe = second_source.fetch_dataframe(
        """
        SELECT
            region,
            revenue
        FROM sales
        """
    )

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["region"] == "West"
    assert dataframe.iloc[0]["revenue"] == 700.0

    second_source.close()


def test_read_only_database_can_read_existing_database(
    tmp_path: Path,
) -> None:
    """A read-only datasource should be able to query an existing database."""

    database_path = (
        tmp_path
        / "datapilot.duckdb"
    )

    writable_source = DuckDBDataSource(
        database_path=database_path,
    )

    writable_source.execute(
        """
        CREATE TABLE sales (
            region VARCHAR,
            revenue DOUBLE
        )
        """
    )

    writable_source.execute(
        """
        INSERT INTO sales
        VALUES ('West', 700)
        """
    )

    writable_source.close()

    read_only_source = DuckDBDataSource(
        database_path=database_path,
        read_only=True,
    )

    dataframe = read_only_source.fetch_dataframe(
        """
        SELECT
            region,
            revenue
        FROM sales
        """
    )

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["region"] == "West"
    assert dataframe.iloc[0]["revenue"] == 700.0

    read_only_source.close()


def test_read_only_database_rejects_writes(
    tmp_path: Path,
) -> None:
    """A read-only datasource should reject write operations."""

    database_path = (
        tmp_path
        / "datapilot.duckdb"
    )

    writable_source = DuckDBDataSource(
        database_path=database_path,
    )

    writable_source.execute(
        """
        CREATE TABLE sales (
            region VARCHAR,
            revenue DOUBLE
        )
        """
    )

    writable_source.close()

    read_only_source = DuckDBDataSource(
        database_path=database_path,
        read_only=True,
    )

    with pytest.raises(Exception):
        read_only_source.execute(
            """
            INSERT INTO sales
            VALUES ('East', 200)
            """
        )

    read_only_source.close()


def test_datasource_can_reconnect_after_close() -> None:
    """A datasource should be reusable after its connection is closed."""

    data_source = DuckDBDataSource()

    first_result = data_source.fetch_dataframe(
        "SELECT 1 AS value"
    )

    assert first_result.iloc[0]["value"] == 1

    data_source.close()

    assert data_source._connection is None

    second_result = data_source.fetch_dataframe(
        "SELECT 2 AS value"
    )

    assert second_result.iloc[0]["value"] == 2

    data_source.close()