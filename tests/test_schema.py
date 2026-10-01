from __future__ import annotations

import pytest

from datapilot.data.duckdb import DuckDBDataSource
from datapilot.data.schema import (
    ColumnInfo,
    ColumnSchema,
    SchemaInspector,
    TableInfo,
    TableSchema,
)
from datapilot.sql.policy import SQLSecurityPolicy


def build_test_database() -> DuckDBDataSource:
    """Create a deterministic database for schema tests."""

    data_source = DuckDBDataSource(":memory:")

    data_source.execute(
        """
        CREATE TABLE sales (
            sale_id INTEGER NOT NULL,
            region VARCHAR NOT NULL,
            revenue DOUBLE,
            customer_id INTEGER
        )
        """
    )

    data_source.execute(
        """
        CREATE TABLE customers (
            customer_id INTEGER NOT NULL,
            customer_name VARCHAR,
            region VARCHAR
        )
        """
    )

    return data_source


def test_list_tables_returns_base_tables_in_sorted_order() -> None:
    """SchemaInspector should return user tables in deterministic order."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    assert inspector.list_tables() == [
        "customers",
        "sales",
    ]

    database.close()


def test_describe_table_returns_table_metadata() -> None:
    """describe_table should return the complete table definition."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    table = inspector.describe_table("sales")

    assert isinstance(table, TableInfo)
    assert table.name == "sales"

    assert table.columns == (
        ColumnInfo(
            name="sale_id",
            data_type="INTEGER",
            nullable=False,
        ),
        ColumnInfo(
            name="region",
            data_type="VARCHAR",
            nullable=False,
        ),
        ColumnInfo(
            name="revenue",
            data_type="DOUBLE",
            nullable=True,
        ),
        ColumnInfo(
            name="customer_id",
            data_type="INTEGER",
            nullable=True,
        ),
    )

    database.close()


def test_describe_table_preserves_column_order() -> None:
    """Columns should follow the database ordinal position."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    table = inspector.describe_table("customers")

    assert [
        column.name
        for column in table.columns
    ] == [
        "customer_id",
        "customer_name",
        "region",
    ]

    database.close()


def test_describe_table_rejects_empty_name() -> None:
    """An empty table name should be rejected."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    with pytest.raises(
        ValueError,
        match="Table name cannot be empty",
    ):
        inspector.describe_table("   ")

    database.close()


def test_describe_table_rejects_unknown_table() -> None:
    """Unknown tables should produce a clear validation error."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    with pytest.raises(
        ValueError,
        match="Table does not exist: missing_table",
    ):
        inspector.describe_table("missing_table")

    database.close()


def test_inspect_returns_all_tables() -> None:
    """inspect should return metadata for every available table."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    tables = inspector.inspect()

    assert isinstance(tables, tuple)
    assert [table.name for table in tables] == [
        "customers",
        "sales",
    ]

    database.close()


def test_inspect_returns_column_metadata_for_each_table() -> None:
    """Every inspected table should contain its column metadata."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    tables = inspector.inspect()

    table_map = {
        table.name: table
        for table in tables
    }

    assert {
        column.name
        for column in table_map["sales"].columns
    } == {
        "sale_id",
        "region",
        "revenue",
        "customer_id",
    }

    assert {
        column.name
        for column in table_map["customers"].columns
    } == {
        "customer_id",
        "customer_name",
        "region",
    }

    database.close()


def test_schema_compatibility_aliases() -> None:
    """Compatibility schema names should reference the production types."""

    assert ColumnSchema is ColumnInfo
    assert TableSchema is TableInfo


def test_security_policy_from_schema() -> None:
    """A policy should authorize tables and columns discovered by the schema."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    policy = SQLSecurityPolicy.from_schema(
        inspector
    )

    assert policy.allowed_tables == frozenset(
        {
            "sales",
            "customers",
        }
    )

    assert policy.allowed_columns["sales"] == frozenset(
        {
            "sale_id",
            "region",
            "revenue",
            "customer_id",
        }
    )

    assert policy.allowed_columns["customers"] == frozenset(
        {
            "customer_id",
            "customer_name",
            "region",
        }
    )

    database.close()


def test_security_policy_allows_known_table() -> None:
    """Known tables should be authorized."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {"region", "revenue"}
            )
        },
    )

    assert policy.is_table_allowed("sales")


def test_security_policy_rejects_unknown_table() -> None:
    """Unknown tables should not be authorized."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {"region", "revenue"}
            )
        },
    )

    assert not policy.is_table_allowed(
        "customers"
    )


def test_security_policy_allows_known_column() -> None:
    """Known columns on authorized tables should be allowed."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {"region", "revenue"}
            )
        },
    )

    assert policy.is_column_allowed(
        "sales",
        "region",
    )

    assert policy.is_column_allowed(
        "sales",
        "revenue",
    )


def test_security_policy_rejects_unknown_column() -> None:
    """Unknown columns should not be authorized."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {"region", "revenue"}
            )
        },
    )

    assert not policy.is_column_allowed(
        "sales",
        "customer_name",
    )


def test_security_policy_rejects_column_on_unknown_table() -> None:
    """Columns on unknown tables should not be authorized."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(
            {"sales"}
        ),
        allowed_columns={
            "sales": frozenset(
                {"region", "revenue"}
            )
        },
    )

    assert not policy.is_column_allowed(
        "customers",
        "region",
    )


def test_security_policy_default_max_result_rows() -> None:
    """The default result limit should be 10,000 rows."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(),
        allowed_columns={},
    )

    assert policy.max_result_rows == 10_000


def test_security_policy_accepts_custom_result_limit() -> None:
    """A positive custom result limit should be accepted."""

    policy = SQLSecurityPolicy(
        allowed_tables=frozenset(),
        allowed_columns={},
        max_result_rows=500,
    )

    assert policy.max_result_rows == 500


@pytest.mark.parametrize(
    "max_result_rows",
    [
        0,
        -1,
        -100,
    ],
)
def test_security_policy_rejects_non_positive_result_limit(
    max_result_rows: int,
) -> None:
    """Result limits must always be positive."""

    with pytest.raises(
        ValueError,
        match="max_result_rows must be greater than zero",
    ):
        SQLSecurityPolicy(
            allowed_tables=frozenset(),
            allowed_columns={},
            max_result_rows=max_result_rows,
        )


def test_security_policy_from_schema_preserves_custom_limit() -> None:
    """from_schema should preserve a configured result limit."""

    database = build_test_database()
    inspector = SchemaInspector(database)

    policy = SQLSecurityPolicy.from_schema(
        inspector,
        max_result_rows=250,
    )

    assert policy.max_result_rows == 250

    database.close()