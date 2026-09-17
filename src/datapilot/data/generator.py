from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for synthetic DataPilot data."""

    seed: int = 42

    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"

    customers: int = 1_000
    products: int = 200
    employees: int = 100
    locations: int = 12

    orders: int = 25_000

    output_dir: str = "data/sample"


class SyntheticDataGenerator:
    """Generate a reproducible synthetic enterprise sales dataset."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> dict[str, pd.DataFrame]:
        """Generate all analytical tables."""

        dim_date = self._generate_date_dimension()
        dim_location = self._generate_location_dimension()
        dim_department = self._generate_department_dimension()
        dim_customer = self._generate_customer_dimension(
            dim_location
        )
        dim_product = self._generate_product_dimension()
        dim_employee = self._generate_employee_dimension(
            dim_department,
            dim_location,
        )

        fact_orders = self._generate_orders(
            dim_date,
            dim_customer,
            dim_employee,
            dim_location,
        )

        fact_sales = self._generate_sales(
            fact_orders,
            dim_product,
        )

        fact_returns = self._generate_returns(
            fact_sales,
            dim_customer,
            dim_product,
            dim_location,
        )

        fact_targets = self._generate_targets(
            dim_date,
            dim_department,
            dim_location,
        )

        return {
            "dim_date": dim_date,
            "dim_customer": dim_customer,
            "dim_product": dim_product,
            "dim_employee": dim_employee,
            "dim_department": dim_department,
            "dim_location": dim_location,
            "fact_sales": fact_sales,
            "fact_orders": fact_orders,
            "fact_returns": fact_returns,
            "fact_targets": fact_targets,
        }

    def save_csv(
        self,
        tables: dict[str, pd.DataFrame],
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save generated tables as CSV files."""

        destination = Path(
            output_dir or self.config.output_dir
        )
        destination.mkdir(parents=True, exist_ok=True)

        for table_name, dataframe in tables.items():
            dataframe.to_csv(
                destination / f"{table_name}.csv",
                index=False,
            )

        return destination

    def _generate_date_dimension(self) -> pd.DataFrame:
        dates = pd.date_range(
            self.config.start_date,
            self.config.end_date,
            freq="D",
        )

        return pd.DataFrame(
            {
                "date_key": dates.strftime("%Y%m%d").astype(int),
                "date": dates,
                "year": dates.year,
                "quarter": dates.quarter,
                "quarter_name": "Q" + dates.quarter.astype(str),
                "month": dates.month,
                "month_name": dates.month_name(),
                "week": dates.isocalendar().week.astype(int),
                "day": dates.day,
                "day_name": dates.day_name(),
                "is_weekend": dates.dayofweek >= 5,
            }
        )

    def _generate_location_dimension(self) -> pd.DataFrame:
        regions = [
            ("North", "India", "Delhi", "Delhi"),
            ("North", "India", "Punjab", "Chandigarh"),
            ("West", "India", "Gujarat", "Ahmedabad"),
            ("West", "India", "Maharashtra", "Mumbai"),
            ("West", "India", "Rajasthan", "Jaipur"),
            ("South", "India", "Karnataka", "Bengaluru"),
            ("South", "India", "Tamil Nadu", "Chennai"),
            ("South", "India", "Telangana", "Hyderabad"),
            ("East", "India", "West Bengal", "Kolkata"),
            ("East", "India", "Odisha", "Bhubaneswar"),
            ("Central", "India", "Madhya Pradesh", "Indore"),
            ("Central", "India", "Madhya Pradesh", "Bhopal"),
        ]

        rows = []

        for index, (
            region,
            country,
            state,
            city,
        ) in enumerate(regions[: self.config.locations], start=1):
            rows.append(
                {
                    "location_key": index,
                    "location_id": f"LOC-{index:03d}",
                    "region": region,
                    "country": country,
                    "state": state,
                    "city": city,
                    "market": (
                        "Metro"
                        if city
                        in {
                            "Delhi",
                            "Mumbai",
                            "Bengaluru",
                            "Chennai",
                            "Hyderabad",
                            "Kolkata",
                        }
                        else "Growth"
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _generate_department_dimension(self) -> pd.DataFrame:
        departments = [
            ("Sales", "Commercial"),
            ("Enterprise Sales", "Commercial"),
            ("Retail", "Commercial"),
            ("Operations", "Operations"),
            ("Finance", "Corporate"),
            ("Customer Success", "Customer"),
        ]

        return pd.DataFrame(
            [
                {
                    "department_key": index,
                    "department_id": f"DEPT-{index:03d}",
                    "department_name": name,
                    "department_group": group,
                }
                for index, (name, group) in enumerate(
                    departments,
                    start=1,
                )
            ]
        )

    def _generate_customer_dimension(
        self,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        segments = [
            "Enterprise",
            "Mid-Market",
            "SMB",
            "Consumer",
        ]

        industries = [
            "Technology",
            "Financial Services",
            "Healthcare",
            "Manufacturing",
            "Retail",
            "Education",
            "Telecommunications",
        ]

        location_keys = locations["location_key"].to_numpy()

        rows = []

        for key in range(1, self.config.customers + 1):
            location_key = int(
                self.rng.choice(location_keys)
            )

            location = locations.loc[
                locations["location_key"] == location_key
            ].iloc[0]

            rows.append(
                {
                    "customer_key": key,
                    "customer_id": f"CUST-{key:05d}",
                    "customer_name": f"Customer {key:05d}",
                    "customer_segment": self.rng.choice(
                        segments,
                        p=[0.15, 0.30, 0.35, 0.20],
                    ),
                    "industry": self.rng.choice(industries),
                    "city": location["city"],
                    "state": location["state"],
                    "country": location["country"],
                    "location_key": location_key,
                }
            )

        return pd.DataFrame(rows)

    def _generate_product_dimension(self) -> pd.DataFrame:
        categories = {
            "Technology": [
                "Laptop",
                "Monitor",
                "Server",
                "Networking",
            ],
            "Software": [
                "Analytics",
                "Security",
                "Productivity",
                "Database",
            ],
            "Services": [
                "Consulting",
                "Implementation",
                "Support",
                "Training",
            ],
            "Hardware": [
                "Printer",
                "Storage",
                "Accessories",
                "Device",
            ],
        }

        brands = [
            "Apex",
            "Nova",
            "Vertex",
            "Nexus",
            "Orion",
        ]

        rows = []

        category_names = list(categories.keys())

        for key in range(1, self.config.products + 1):
            category = self.rng.choice(category_names)

            subcategory = self.rng.choice(
                categories[category]
            )

            unit_cost = float(
                self.rng.uniform(50, 5_000)
            )

            list_price = unit_cost * float(
                self.rng.uniform(1.20, 2.20)
            )

            rows.append(
                {
                    "product_key": key,
                    "product_id": f"PROD-{key:05d}",
                    "product_name": (
                        f"{subcategory} Product {key:04d}"
                    ),
                    "category": category,
                    "subcategory": subcategory,
                    "brand": self.rng.choice(brands),
                    "unit_cost": round(unit_cost, 2),
                    "list_price": round(list_price, 2),
                }
            )

        return pd.DataFrame(rows)

    def _generate_employee_dimension(
        self,
        departments: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        job_titles = [
            "Sales Executive",
            "Account Manager",
            "Sales Manager",
            "Business Analyst",
            "Operations Manager",
            "Customer Success Manager",
        ]

        rows = []

        for key in range(1, self.config.employees + 1):
            rows.append(
                {
                    "employee_key": key,
                    "employee_id": f"EMP-{key:04d}",
                    "employee_name": f"Employee {key:04d}",
                    "job_title": self.rng.choice(job_titles),
                    "department_key": int(
                        self.rng.choice(
                            departments["department_key"].to_numpy()
                        )
                    ),
                    "location_key": int(
                        self.rng.choice(
                            locations["location_key"].to_numpy()
                        )
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _generate_orders(
        self,
        dates: pd.DataFrame,
        customers: pd.DataFrame,
        employees: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        date_keys = dates["date_key"].to_numpy()
        customer_keys = customers["customer_key"].to_numpy()
        employee_keys = employees["employee_key"].to_numpy()
        location_keys = locations["location_key"].to_numpy()

        order_keys = np.arange(
            1,
            self.config.orders + 1,
            dtype=np.int64,
        )

        selected_dates = self.rng.choice(
            date_keys,
            size=self.config.orders,
        )

        selected_customers = self.rng.choice(
            customer_keys,
            size=self.config.orders,
        )

        selected_employees = self.rng.choice(
            employee_keys,
            size=self.config.orders,
        )

        selected_locations = self.rng.choice(
            location_keys,
            size=self.config.orders,
        )

        item_counts = self.rng.integers(
            1,
            6,
            size=self.config.orders,
        )

        order_amounts = np.round(
            self.rng.lognormal(
                mean=7.4,
                sigma=0.8,
                size=self.config.orders,
            ),
            2,
        )

        statuses = self.rng.choice(
            [
                "Completed",
                "Completed",
                "Completed",
                "Pending",
                "Cancelled",
            ],
            size=self.config.orders,
        )

        shipping = np.round(
            self.rng.uniform(
                0,
                500,
                size=self.config.orders,
            ),
            2,
        )

        return pd.DataFrame(
            {
                "order_key": order_keys,
                "date_key": selected_dates,
                "customer_key": selected_customers,
                "employee_key": selected_employees,
                "location_key": selected_locations,
                "order_status": statuses,
                "order_amount": order_amounts,
                "item_count": item_counts,
                "shipping_amount": shipping,
            }
        )

    def _generate_sales(
        self,
        orders: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        product_keys = products["product_key"].to_numpy()

        sales_key = 1

        for order in orders.itertuples(index=False):
            product_choices = self.rng.choice(
                product_keys,
                size=order.item_count,
                replace=False,
            )

            for product_key in product_choices:
                product = products.loc[
                    products["product_key"] == product_key
                ].iloc[0]

                quantity = int(
                    self.rng.integers(1, 8)
                )

                unit_price = float(product["list_price"])

                # Seasonal effect.
                month = (
                    int(str(order.date_key)[4:6])
                    if len(str(order.date_key)) == 8
                    else 1
                )

                seasonal_factor = (
                    1.15
                    if month in {10, 11, 12}
                    else 1.00
                )

                # Regional variation.
                regional_factor = float(
                    self.rng.uniform(0.85, 1.20)
                )

                unit_price *= (
                    seasonal_factor * regional_factor
                )

                discount_rate = float(
                    self.rng.uniform(0.00, 0.25)
                )

                gross_amount = (
                    quantity * unit_price
                )

                discount_amount = (
                    gross_amount * discount_rate
                )

                net_amount = (
                    gross_amount - discount_amount
                )

                cost_amount = (
                    quantity * float(product["unit_cost"])
                )

                profit_amount = (
                    net_amount - cost_amount
                )

                rows.append(
                    {
                        "sales_key": sales_key,
                        "date_key": order.date_key,
                        "customer_key": order.customer_key,
                        "product_key": product_key,
                        "employee_key": order.employee_key,
                        "location_key": order.location_key,
                        "order_key": order.order_key,
                        "quantity": quantity,
                        "unit_price": round(unit_price, 2),
                        "gross_amount": round(
                            gross_amount,
                            2,
                        ),
                        "discount_amount": round(
                            discount_amount,
                            2,
                        ),
                        "net_amount": round(
                            net_amount,
                            2,
                        ),
                        "cost_amount": round(
                            cost_amount,
                            2,
                        ),
                        "profit_amount": round(
                            profit_amount,
                            2,
                        ),
                        "currency": "INR",
                    }
                )

                sales_key += 1

        sales = pd.DataFrame(rows)

        # Introduce a small number of missing values.
        missing_mask = (
            self.rng.random(len(sales)) < 0.002
        )

        sales.loc[
            missing_mask,
            "discount_amount",
        ] = np.nan

        # Introduce controlled outliers.
        outlier_mask = (
            self.rng.random(len(sales)) < 0.001
        )

        sales.loc[
            outlier_mask,
            "net_amount",
        ] *= 5

        return sales

    def _generate_returns(
        self,
        sales: pd.DataFrame,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        if sales.empty:
            return pd.DataFrame(
                columns=[
                    "return_key",
                    "date_key",
                    "sales_key",
                    "order_key",
                    "customer_key",
                    "product_key",
                    "location_key",
                    "return_quantity",
                    "return_amount",
                    "return_reason",
                ]
            )

        return_probability = 0.06

        mask = (
            self.rng.random(len(sales))
            < return_probability
        )

        selected = sales.loc[mask].copy()

        reasons = [
            "Damaged",
            "Incorrect Product",
            "Customer Changed Mind",
            "Quality Issue",
            "Duplicate Order",
        ]

        selected["return_quantity"] = [
            int(
                self.rng.integers(
                    1,
                    max(2, int(quantity) + 1),
                )
            )
            for quantity in selected["quantity"]
        ]

        selected["return_amount"] = (
            selected["net_amount"]
            * (
                selected["return_quantity"]
                / selected["quantity"]
            )
        ).round(2)

        selected["return_reason"] = self.rng.choice(
            reasons,
            size=len(selected),
        )

        selected["return_key"] = np.arange(
            1,
            len(selected) + 1,
            dtype=np.int64,
        )

        return selected[
            [
                "return_key",
                "date_key",
                "sales_key",
                "order_key",
                "customer_key",
                "product_key",
                "location_key",
                "return_quantity",
                "return_amount",
                "return_reason",
            ]
        ].reset_index(drop=True)

    def _generate_targets(
        self,
        dates: pd.DataFrame,
        departments: pd.DataFrame,
        locations: pd.DataFrame,
    ) -> pd.DataFrame:
        monthly_dates = (
            dates[
                [
                    "date_key",
                    "year",
                    "month",
                ]
            ]
            .drop_duplicates(
                subset=["year", "month"]
            )
            .sort_values(
                ["year", "month"]
            )
        )

        rows = []

        target_key = 1

        for date in monthly_dates.itertuples(index=False):
            for location_key in locations["location_key"]:
                for department_key in departments[
                    "department_key"
                ]:
                    target_amount = float(
                        self.rng.uniform(
                            500_000,
                            2_500_000,
                        )
                    )

                    rows.append(
                        {
                            "target_key": target_key,
                            "date_key": date.date_key,
                            "location_key": int(location_key),
                            "department_key": int(
                                department_key
                            ),
                            "target_type": "Revenue",
                            "target_amount": round(
                                target_amount,
                                2,
                            ),
                        }
                    )

                    target_key += 1

        return pd.DataFrame(rows)


def generate_dataset(
    config: DatasetConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience function for generating the full dataset."""

    generator = SyntheticDataGenerator(config)
    return generator.generate()