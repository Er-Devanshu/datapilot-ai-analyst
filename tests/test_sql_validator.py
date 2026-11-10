
from datapilot.sql.validator import SQLValidator


def test_valid_read_only_query() -> None:
    validator = SQLValidator()

    result = validator.validate(
        """
        SELECT
            d.region,
            SUM(s.net_amount) AS revenue
        FROM fact_sales AS s
        JOIN dim_location AS d
            ON s.location_key = d.location_key
        GROUP BY d.region
        ORDER BY revenue DESC
        """,
        allowed_tables={
            "fact_sales",
            "dim_location",
        },
        allowed_columns={
            "fact_sales": {
                "location_key",
                "net_amount",
            },
            "dim_location": {
                "location_key",
                "region",
            },
        },
    )

    assert result.valid is True
    assert result.errors == ()
    assert "ORDER BY revenue DESC" in result.sql
    assert "LIMIT 10001" in result.sql


def test_write_statement_is_rejected() -> None:
    validator = SQLValidator()

    result = validator.validate(
        "DROP TABLE fact_sales",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is False


def test_unauthorized_table_is_rejected() -> None:
    validator = SQLValidator()

    result = validator.validate(
        "SELECT * FROM secret_table",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is False
    assert any(
        "Unauthorized tables" in error
        for error in result.errors
    )


def test_unauthorized_column_is_rejected() -> None:
    validator = SQLValidator()

    result = validator.validate(
        "SELECT s.password FROM fact_sales AS s",
        allowed_tables={"fact_sales"},
        allowed_columns={
            "fact_sales": {
                "sales_key",
                "net_amount",
            }
        },
    )

    assert result.valid is False
    assert any(
        "columns" in error.lower()
        for error in result.errors
    )


def test_result_limit_is_enforced() -> None:
    validator = SQLValidator(
        max_result_rows=10_000
    )

    result = validator.validate(
        "SELECT * FROM fact_sales LIMIT 50000",
        allowed_tables={"fact_sales"},
        allowed_columns={
            "fact_sales": {
                "sales_key",
            }
        },
    )

    assert result.valid is False
    assert any(
        "limit" in error.lower()
        for error in result.errors
    )


def test_unbounded_query_gets_sentinel_limit() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        "SELECT * FROM fact_sales",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is True
    assert "LIMIT 101" in result.sql


def test_automatic_limit_preserves_order_by() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        """
        SELECT region, net_amount
        FROM fact_sales
        ORDER BY net_amount DESC
        """,
        allowed_tables={"fact_sales"},
    )

    assert result.valid is True
    assert "ORDER BY net_amount DESC" in result.sql
    assert result.sql.endswith("LIMIT 101")


def test_explicit_smaller_limit_is_preserved() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        "SELECT * FROM fact_sales LIMIT 25",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is True
    assert result.sql.endswith("LIMIT 25")
    assert "LIMIT 101" not in result.sql


def test_explicit_limit_equal_to_maximum_is_preserved() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        "SELECT * FROM fact_sales LIMIT 100",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is True
    assert result.sql.endswith("LIMIT 100")


def test_dynamic_limit_is_rejected() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        """
        SELECT *
        FROM fact_sales
        LIMIT (SELECT COUNT(*) FROM fact_sales)
        """,
        allowed_tables={"fact_sales"},
    )

    assert result.valid is False
    assert any(
        "Dynamic or unsupported LIMIT" in error
        for error in result.errors
    )


def test_fractional_limit_is_rejected() -> None:
    validator = SQLValidator(
        max_result_rows=100
    )

    result = validator.validate(
        "SELECT * FROM fact_sales LIMIT 10.5",
        allowed_tables={"fact_sales"},
    )

    assert result.valid is False


def test_multiple_statements_are_rejected() -> None:
    validator = SQLValidator()

    result = validator.validate(
        """
        SELECT * FROM fact_sales;
        DROP TABLE fact_sales;
        """,
        allowed_tables={"fact_sales"},
    )

    assert result.valid is False


def test_empty_sql_is_rejected() -> None:
    validator = SQLValidator()

    result = validator.validate("")

    assert result.valid is False
