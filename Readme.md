# DataPilot — AI Data Analyst Agent

> **Ask questions. Explore data. Discover insights.**
>
> DataPilot is an open-source, agentic AI Data Analyst designed to understand natural-language business questions, reason over structured data, generate and validate SQL, perform multi-step analysis, create visualizations, and return evidence-backed insights.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://www.langchain.com/langgraph)
[![DuckDB](https://img.shields.io/badge/Analytics-DuckDB-FFF000)](https://duckdb.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/UI-Next.js-000000?logo=next.js\&logoColor=white)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Traditional BI tools require users to understand dashboards, filters, dimensions, measures, and reporting structures.

DataPilot explores a different interaction model:

```text
                Natural Language
                       │
                       ▼
                ┌─────────────┐
                │   Planner   │
                └──────┬──────┘
                       │
                       ▼
              Schema & Metadata
                  Discovery
                       │
                       ▼
                Semantic Layer
                       │
                       ▼
                SQL Generation
                       │
                       ▼
                SQL Validation
                       │
                  ┌────┴────┐
                  │         │
               Invalid    Valid
                  │         │
                  ▼         ▼
               Repair    Execute
                  │         │
                  └────┬────┘
                       │
                       ▼
                 Data Analysis
                       │
               ┌───────┼───────┐
               ▼       ▼       ▼
            Trends  Drivers  Segments
               │       │       │
               └───────┼───────┘
                       ▼
                 Visualization
                       │
                       ▼
                Evidence-backed
                    Insight
```

The goal is not simply to generate SQL.

The goal is to build a system capable of behaving like a **data analyst**.

---

# Why DataPilot?

Large Language Models can translate natural language into SQL, but generating syntactically valid SQL is only one part of analytical reasoning.

A useful AI Data Analyst must also understand:

* business terminology
* database schemas
* relationships between tables
* metric definitions
* time periods
* analytical intent
* query correctness
* data quality
* statistical context
* visualization requirements
* uncertainty and limitations

DataPilot is designed around these requirements.

---

# Example

A user can ask:

> **Why did revenue decline in July compared with June?**

Instead of generating one SQL query and stopping, DataPilot can decompose the problem:

```text
1. Determine the comparison periods
2. Calculate June revenue
3. Calculate July revenue
4. Quantify the overall variance
5. Analyze variance by region
6. Analyze variance by product category
7. Identify the largest contributors
8. Validate the findings
9. Generate supporting visualizations
10. Produce a concise business explanation
```

The resulting response can contain:

```text
┌───────────────────────────────────────────┐
│ INSIGHT                                   │
│                                           │
│ Revenue declined 12.7% in July compared  │
│ with June. The largest contributors were │
│ the Enterprise segment and West region.  │
└───────────────────────────────────────────┘

[ Visualization ]

SQL
└── View generated query

Evidence
└── View supporting data

Agent Trace
└── View analytical workflow
```

---

# Core Capabilities

DataPilot is being developed around the following capabilities.

## Natural Language Analytics

Ask questions using normal business language.

Examples:

```text
What was our total revenue in Q2?

Which products generated the most revenue?

Which regions grew fastest this year?

Compare Q1 and Q2 revenue.

What was our year-over-year growth?
```

## Multi-Step Analytical Reasoning

Complex questions can be decomposed into multiple analytical tasks.

```text
Question
   ↓
Planning
   ↓
Multiple analytical queries
   ↓
Result synthesis
   ↓
Business insight
```

## Text-to-SQL

Convert natural-language questions into executable SQL using schema and semantic context.

## SQL Validation

Generated SQL is validated before execution.

DataPilot is designed to prevent arbitrary database modification and unsafe query execution.

## SQL Repair

When a generated query fails, the system can inspect the error, repair the query, validate it again, and retry within controlled limits.

## Semantic Layer

Business metrics and dimensions are explicitly defined rather than allowing the model to guess their meaning.

Example:

```yaml
metrics:
  revenue:
    expression: SUM(fact_sales.net_amount)
    description: Net sales revenue excluding returns

dimensions:
  region:
    column: dim_location.region

  product:
    column: dim_product.product_name
```

## Visualization

Analytical results can be transformed into structured visualization specifications and rendered as charts.

## Conversational Analytics

Users can ask follow-up questions while retaining relevant analytical context.

```text
User:
Show revenue by region.

DataPilot:
[Chart]

User:
Only regions above ₹1M.

DataPilot:
[Updated Chart]

User:
Compare those with last year.

DataPilot:
[Comparison]
```

## Evidence & Transparency

DataPilot is designed to expose supporting information such as:

* generated SQL
* query results
* analytical steps
* charts
* assumptions
* errors and retries

The objective is to make AI-generated analytics more inspectable.

---

# Architecture

The target architecture is:

```text
                         ┌──────────────────────┐
                         │      User / UI       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      LangGraph       │
                         │   Agent Runtime      │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┼─────────────────────┐
             │                      │                     │
             ▼                      ▼                     ▼
       Schema Agent          SQL Generator          Analyzer
             │                      │                     │
             └──────────────────────┼─────────────────────┘
                                    │
                                    ▼
                             SQL Validation
                                    │
                               ┌────┴────┐
                               ▼         ▼
                            Repair    Execute
                               │         │
                               └────┬────┘
                                    ▼
                              Visualization
                                    │
                                    ▼
                              Answer Composer
```

---

# Technology Stack

DataPilot is intentionally designed to use a **zero-cost, open-source development stack**.

| Layer                   | Technology                      |
| ----------------------- | ------------------------------- |
| Language                | Python                          |
| Agent orchestration     | LangGraph                       |
| LLM inference           | Open-source LLMs                |
| Development environment | Google Colab                    |
| Analytics engine        | DuckDB                          |
| Data processing         | Pandas / Polars                 |
| SQL parsing             | SQLGlot                         |
| API                     | FastAPI                         |
| UI                      | Gradio initially                |
| Visualization           | Plotly                          |
| Vector search           | FAISS / ChromaDB where required |
| Testing                 | Pytest                          |
| Version control         | Git / GitHub                    |
| Containerization        | Docker                          |
| CI                      | GitHub Actions                  |

The initial development workflow is designed to run without paid API services or cloud infrastructure.

---

# Development Environment

The project is being developed using **Google Colab** to maintain a zero-cost development workflow.

The initial architecture is therefore notebook-friendly while keeping the codebase structured so it can later be packaged as a conventional Python application.

```text
Google Colab
     │
     ├── Python
     ├── Open-source LLM
     ├── LangGraph
     ├── DuckDB
     ├── SQLGlot
     └── DataPilot
```

---

# Dataset

DataPilot will initially use a synthetic enterprise sales dataset.

The dataset will contain realistic analytical entities such as:

```text
dim_date
dim_customer
dim_product
dim_employee
dim_department
dim_location

fact_sales
fact_orders
fact_returns
fact_targets
```

The data will include realistic characteristics such as:

* multiple business dimensions
* time-series behavior
* regional differences
* product categories
* customer segments
* targets
* discounts
* returns
* missing values
* outliers
* seasonality

No proprietary or client data will be required.

---

# Security & Guardrails

LLM-generated SQL should never be treated as inherently trustworthy.

DataPilot therefore includes a security layer between SQL generation and execution.

The intended execution pipeline is:

```text
LLM-generated SQL
       │
       ▼
   SQL Parser
       │
       ▼
Security Validation
       │
       ▼
Read-only enforcement
       │
       ▼
Allowed tables / columns
       │
       ▼
Query limits
       │
       ▼
Database execution
```

The system will explicitly defend against:

* destructive SQL
* unauthorized tables
* unsafe queries
* excessive result sizes
* uncontrolled agent loops
* prompt injection through data
* fabricated schema elements

---

# Evaluation

A major goal of DataPilot is to evaluate the system rather than simply demonstrate it.

The project will maintain an analytical benchmark containing questions with expected semantic outcomes.

Evaluation areas include:

## SQL

* SQL validity
* execution success
* semantic correctness

## Analytics

* numerical accuracy
* metric correctness
* filtering correctness
* time-period correctness

## Generation

* answer relevance
* groundedness
* factual consistency

## Agent

* successful task completion
* retry behavior
* failure handling

## Performance

* latency
* model usage
* query execution time

Example evaluation report:

```text
DataPilot Evaluation
────────────────────────────

Questions evaluated          100

SQL validity                  XX%
Execution success             XX%
Semantic accuracy             XX%
Answer accuracy               XX%

Average latency              X.Xs
Average SQL retries           X.X
```

> Evaluation numbers will be generated from actual experiments and will not be hard-coded or fabricated.

---

# Project Structure

The repository will evolve toward:

```text
datapilot-ai-analyst/
│
├── notebooks/
│   ├── 01_data_platform.ipynb
│   ├── 02_text_to
```
