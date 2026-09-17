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