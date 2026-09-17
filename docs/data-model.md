# DataPilot Analytical Data Model

## 1. Purpose

DataPilot uses a synthetic enterprise sales dataset designed to support
natural-language business analytics, text-to-SQL, analytical reasoning,
visualization, and evaluation.

The model follows a dimensional analytical design with:

- Date dimension
- Customer dimension
- Product dimension
- Employee dimension
- Department dimension
- Location dimension
- Sales fact
- Orders fact
- Returns fact
- Targets fact

The model is synthetic and contains no proprietary or client data.

---

## 2. Model Overview

```text
                         dim_date
                            │
                            │
                            ▼
dim_customer ────────── fact_sales ────────── dim_product
                            │
                            │
                            ├──────────── dim_employee
                            │
                            └──────────── dim_location
                                          │
                                          ▼
                                    dim_department


dim_date ───────────── fact_orders
dim_date ───────────── fact_returns
dim_date ───────────── fact_targets