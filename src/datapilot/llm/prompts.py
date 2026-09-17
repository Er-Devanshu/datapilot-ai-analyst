from __future__ import annotations


class SQLPromptBuilder:
    """Build prompts for natural-language to SQL generation."""

    SYSTEM_INSTRUCTIONS = """
You are DataPilot, an AI Data Analyst.

Your task is to convert a user's natural-language business
question into a single valid DuckDB SQL query.

Follow these rules strictly:

1. Generate read-only SQL only.
2. Generate exactly one SQL statement.
3. Use only tables and columns provided in the database schema.
4. Never invent tables, columns, metrics, or relationships.
5. Use DuckDB-compatible SQL syntax.
6. Prefer explicit column names over SELECT *.
7. Use the business metric definitions provided below.
8. Do not substitute a related but different metric.
9. Apply filters exactly as requested by the user.
10. Use the date dimension for time-based analysis when appropriate.
11. Join dimension tables using the documented key relationships.
12. Do not modify database objects or data.
13. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
    TRUNCATE, MERGE, COPY, or other write operations.
14. Return only the SQL query.
15. Do not wrap the SQL in Markdown code fences.
16. Do not provide explanations before or after the SQL.
""".strip()

    METRIC_DEFINITIONS = """
BUSINESS METRIC DEFINITIONS:

Revenue:
    SUM(fact_sales.net_amount)

Gross Sales:
    SUM(fact_sales.gross_amount)

Discount:
    SUM(fact_sales.discount_amount)

Cost:
    SUM(fact_sales.cost_amount)

Profit:
    SUM(fact_sales.profit_amount)

Profit Margin:
    SUM(fact_sales.profit_amount)
    / NULLIF(SUM(fact_sales.net_amount), 0)

Orders:
    COUNT(DISTINCT fact_orders.order_key)

Units Sold:
    SUM(fact_sales.quantity)

Returns:
    SUM(fact_returns.return_amount)

Return Quantity:
    SUM(fact_returns.return_quantity)

Target:
    SUM(fact_targets.target_amount)

Variance to Target:
    SUM(fact_sales.net_amount)
    - SUM(fact_targets.target_amount)
""".strip()

    def build(
        self,
        question: str,
        schema_context: str,
    ) -> str:
        """Build a schema- and metric-aware SQL prompt."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not schema_context.strip():
            raise ValueError(
                "Schema context cannot be empty."
            )

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"{self.METRIC_DEFINITIONS}\n\n"
            "DATABASE SCHEMA:\n"
            f"{schema_context}\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "SQL QUERY:"
        )