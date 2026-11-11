from __future__ import annotations

import json


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
            raise ValueError("Question cannot be empty.")

        if not schema_context.strip():
            raise ValueError("Schema context cannot be empty.")

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"{self.METRIC_DEFINITIONS}\n\n"
            "DATABASE SCHEMA:\n"
            f"{schema_context}\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "SQL QUERY:"
        )


class AnalyticsPlanPromptBuilder:
    """Build structured prompts for deterministic analytics planning."""

    SYSTEM_INSTRUCTIONS = """
You are DataPilot, an AI Data Analyst responsible for proposing
deterministic analytical operations over an already executed query result.

Your task is to translate the user's analytical question into a JSON
analytics plan.

You are a planner, not an executor.

STRICT RULES:

1. Use only columns listed in the supplied query-result metadata.
2. Never invent column names, period values, or business metrics.
3. Select only supported analysis types.
4. Use the smallest set of operations that can answer the question.
5. Do not create operations unrelated to the user's question.
6. Do not calculate or fabricate analytical results.
7. Do not generate SQL, Python, executable code, or tool instructions.
8. Do not include explanations outside the JSON object.
9. Return valid JSON only. Do not use Markdown code fences.
10. Use exact period labels from the supplied temporal values when available.
11. If the question cannot be answered with the available columns,
    return an empty requests array.
12. Do not claim that a relationship is causal.
""".strip()

    OUTPUT_SCHEMA = {
        "requests": [
            {
                "request_id": "unique_string",
                "analysis_type": (
                    "trend | period_comparison | variance | "
                    "contribution | segmentation | root_cause"
                ),
                "inputs": {
                    "period_column": "string or null",
                    "value_column": "string or null",
                    "dimension_column": "string or null",
                    "reference_period": "string or null",
                    "current_period": "string or null",
                    "actual_period": "string or null",
                    "aggregation": (
                        "sum | mean | min | max | count | null"
                    ),
                },
                "depends_on": ["request_id"],
            }
        ]
    }

    ANALYSIS_RULES = """
SUPPORTED ANALYSIS TYPES AND REQUIRED INPUTS:

trend:
- Requires period_column and numeric value_column.
- Does not accept aggregation.
- Does not require dependencies.

period_comparison:
- Requires period_column, value_column, reference_period,
  current_period, and an explicit aggregation.
- Supported aggregations: sum, mean, min, max, count.
- The two period labels must be distinct and available in the data.
- Does not require dependencies.

variance:
- Requires period_column, value_column, reference_period,
  actual_period, and an explicit aggregation.
- Supported aggregations: sum, mean, min, max, count.
- The two period labels must be distinct and available in the data.
- Does not require dependencies.

contribution:
- Requires dimension_column and value_column.
- value_column must be one of the trusted category-level change columns
  supplied by the application.
- Does not accept aggregation.
- Does not require dependencies.

segmentation:
- Requires dimension_column and numeric value_column.
- Does not accept aggregation.
- Does not require dependencies.

root_cause:
- Does not accept direct column mappings.
- Must depend on exactly one variance request and one contribution request.
- Use the IDs of those requests in depends_on.
- This operation reconciles arithmetic contributions; it does not prove
  causality.

AGGREGATION MEANING:
- sum: sum of non-null values.
- mean: arithmetic mean of non-null numeric values.
- min: minimum numeric value.
- max: maximum numeric value.
- count: count of non-null values in the mapped value column.

DEPENDENCIES:
- A dependency must refer to another request in this same plan.
- Do not create missing or duplicated dependency IDs.
- Do not create dependency cycles.
- A root_cause request must depend on a variance and a contribution request.
""".strip()

    def build(
        self,
        question: str,
        dataframe_metadata: dict[str, object],
        change_value_columns: set[str] | frozenset[str] | None = None,
    ) -> str:
        """Build a JSON-only analytics planning prompt."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(dataframe_metadata, dict):
            raise TypeError("Dataframe metadata must be a dictionary.")

        if change_value_columns is None:
            trusted_columns: list[str] = []
        else:
            if not isinstance(change_value_columns, (set, frozenset)):
                raise ValueError(
                    "Change value columns must be a set or frozenset."
                )

            if not all(
                isinstance(column, str) and column.strip()
                for column in change_value_columns
            ):
                raise ValueError(
                    "Trusted change column names must be non-empty strings."
                )

            trusted_columns = sorted(change_value_columns)

        metadata_text = json.dumps(
            dataframe_metadata,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        schema_text = json.dumps(
            self.OUTPUT_SCHEMA,
            indent=2,
            ensure_ascii=False,
        )

        trusted_text = json.dumps(
            trusted_columns,
            ensure_ascii=False,
        )

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"{self.ANALYSIS_RULES}\n\n"
            "REQUIRED JSON OUTPUT SCHEMA:\n"
            f"{schema_text}\n\n"
            "QUERY RESULT METADATA:\n"
            f"{metadata_text}\n\n"
            "TRUSTED CATEGORY-LEVEL CHANGE COLUMNS "
            "(APPLICATION-PROVIDED; DO NOT INFER):\n"
            f"{trusted_text}\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "RETURN THE JSON OBJECT NOW:"
        )