from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityCheck:
    """Result of a single data-quality check."""

    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class QualityReport:
    """Complete data-quality report."""

    checks: tuple[QualityCheck, ...]

    @property
    def passed(self) -> bool:
        """Return True when every quality check passes."""

        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[QualityCheck, ...]:
        """Return all failed checks."""

        return tuple(
            check
            for check in self.checks
            if not check.passed
        )


class DatasetQualityValidator:
    """Validate structural and analytical quality of the dataset."""

    REQUIRED_TABLES = (
        "dim_date",
        "dim_customer",
        "dim_product",
        "dim_employee",
        "dim_department",
        "dim_location",
        "fact_sales",
        "fact_orders",
        "fact_returns",
        "fact_targets",
    )

    def validate(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> QualityReport:
        """Run all quality checks."""

        checks: list[QualityCheck] = []

        checks.append(
            self._check_required_tables(tables)
        )

        if not self._check_passed(
            checks[-1]
        ):
            return QualityReport(tuple(checks))

        checks.extend(
            self._check_primary_keys(tables)
        )

        checks.extend(
            self._check_foreign_keys(tables)
        )

        checks.extend(
            self._check_business_values(tables)
        )

        checks.extend(
            self._check_data_completeness(tables)
        )

        return QualityReport(tuple(checks))

    def _check_required_tables(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> QualityCheck:
        missing = [
            table
            for table in self.REQUIRED_TABLES
            if table not in tables
        ]

        return QualityCheck(
            name="required_tables",
            passed=not missing,
            details=(
                "All required tables are present."
                if not missing
                else f"Missing tables: {missing}"
            ),
        )

    def _check_primary_keys(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> list[QualityCheck]:
        primary_keys = {
            "dim_date": "date_key",
            "dim_customer": "customer_key",
            "dim_product": "product_key",
            "dim_employee": "employee_key",
            "dim_department": "department_key",
            "dim_location": "location_key",
            "fact_sales": "sales_key",
            "fact_orders": "order_key",
            "fact_returns": "return_key",
            "fact_targets": "target_key",
        }

        checks = []

        for table_name, key_column in primary_keys.items():
            dataframe = tables[table_name]

            null_count = int(
                dataframe[key_column].isna().sum()
            )

            duplicate_count = int(
                dataframe[key_column].duplicated().sum()
            )

            passed = (
                null_count == 0
                and duplicate_count == 0
            )

            checks.append(
                QualityCheck(
                    name=f"primary_key:{table_name}",
                    passed=passed,
                    details=(
                        f"{key_column} is unique and non-null."
                        if passed
                        else (
                            f"{key_column}: "
                            f"{null_count} nulls, "
                            f"{duplicate_count} duplicates."
                        )
                    ),
                )
            )

        return checks

    def _check_foreign_keys(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> list[QualityCheck]:
        relationships = (
            (
                "fact_sales",
                "customer_key",
                "dim_customer",
                "customer_key",
            ),
            (
                "fact_sales",
                "product_key",
                "dim_product",
                "product_key",
            ),
            (
                "fact_sales",
                "employee_key",
                "dim_employee",
                "employee_key",
            ),
            (
                "fact_sales",
                "location_key",
                "dim_location",
                "location_key",
            ),
            (
                "fact_sales",
                "date_key",
                "dim_date",
                "date_key",
            ),
            (
                "fact_orders",
                "customer_key",
                "dim_customer",
                "customer_key",
            ),
            (
                "fact_orders",
                "employee_key",
                "dim_employee",
                "employee_key",
            ),
            (
                "fact_orders",
                "location_key",
                "dim_location",
                "location_key",
            ),
            (
                "fact_orders",
                "date_key",
                "dim_date",
                "date_key",
            ),
            (
                "fact_returns",
                "sales_key",
                "fact_sales",
                "sales_key",
            ),
            (
                "fact_returns",
                "order_key",
                "fact_orders",
                "order_key",
            ),
            (
                "fact_returns",
                "customer_key",
                "dim_customer",
                "customer_key",
            ),
            (
                "fact_returns",
                "product_key",
                "dim_product",
                "product_key",
            ),
            (
                "fact_returns",
                "location_key",
                "dim_location",
                "location_key",
            ),
            (
                "fact_returns",
                "date_key",
                "dim_date",
                "date_key",
            ),
            (
                "fact_targets",
                "date_key",
                "dim_date",
                "date_key",
            ),
            (
                "fact_targets",
                "location_key",
                "dim_location",
                "location_key",
            ),
            (
                "fact_targets",
                "department_key",
                "dim_department",
                "department_key",
            ),
        )

        checks = []

        for (
            child_table,
            child_column,
            parent_table,
            parent_column,
        ) in relationships:
            child_values = set(
                tables[child_table][child_column]
                .dropna()
                .unique()
            )

            parent_values = set(
                tables[parent_table][parent_column]
                .dropna()
                .unique()
            )

            orphan_values = child_values - parent_values

            passed = not orphan_values

            checks.append(
                QualityCheck(
                    name=(
                        f"foreign_key:"
                        f"{child_table}.{child_column}"
                    ),
                    passed=passed,
                    details=(
                        "All foreign-key values have "
                        "matching parent records."
                        if passed
                        else (
                            f"{len(orphan_values)} orphan "
                            "foreign-key values found."
                        )
                    ),
                )
            )

        return checks

    def _check_business_values(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> list[QualityCheck]:
        checks = []

        sales = tables["fact_sales"]

        checks.append(
            self._non_negative_check(
                "sales_quantity",
                sales["quantity"],
            )
        )

        checks.append(
            self._non_negative_check(
                "sales_gross_amount",
                sales["gross_amount"],
            )
        )

        checks.append(
            self._non_negative_check(
                "sales_discount_amount",
                sales["discount_amount"],
            )
        )

        checks.append(
            self._non_negative_check(
                "sales_cost_amount",
                sales["cost_amount"],
            )
        )

        orders = tables["fact_orders"]

        checks.append(
            self._non_negative_check(
                "order_amount",
                orders["order_amount"],
            )
        )

        returns = tables["fact_returns"]

        checks.append(
            self._non_negative_check(
                "return_quantity",
                returns["return_quantity"],
            )
        )

        checks.append(
            self._non_negative_check(
                "return_amount",
                returns["return_amount"],
            )
        )

        targets = tables["fact_targets"]

        checks.append(
            self._non_negative_check(
                "target_amount",
                targets["target_amount"],
            )
        )

        return checks

    def _check_data_completeness(
        self,
        tables: dict[str, pd.DataFrame],
    ) -> list[QualityCheck]:
        checks = []

        sales = tables["fact_sales"]

        required_sales_columns = (
            "sales_key",
            "date_key",
            "customer_key",
            "product_key",
            "employee_key",
            "location_key",
            "order_key",
            "quantity",
            "unit_price",
            "gross_amount",
            "net_amount",
            "cost_amount",
            "profit_amount",
            "currency",
        )

        null_counts = {
            column: int(
                sales[column].isna().sum()
            )
            for column in required_sales_columns
            if column in sales.columns
        }

        unexpected_nulls = {
            column: count
            for column, count in null_counts.items()
            if count > 0
        }

        # discount_amount is intentionally allowed to contain
        # missing values because the dataset specification includes
        # missing-value scenarios.
        passed = not unexpected_nulls

        checks.append(
            QualityCheck(
                name="sales_required_fields",
                passed=passed,
                details=(
                    "Required sales fields contain no nulls."
                    if passed
                    else (
                        "Unexpected nulls: "
                        f"{unexpected_nulls}"
                    )
                ),
            )
        )

        return checks

    @staticmethod
    def _non_negative_check(
        name: str,
        series: pd.Series,
    ) -> QualityCheck:
        negative_count = int(
            (series.dropna() < 0).sum()
        )

        return QualityCheck(
            name=f"non_negative:{name}",
            passed=negative_count == 0,
            details=(
                "No negative values found."
                if negative_count == 0
                else (
                    f"{negative_count} negative "
                    "values found."
                )
            ),
        )

    @staticmethod
    def _check_passed(
        check: QualityCheck,
    ) -> bool:
        return check.passed