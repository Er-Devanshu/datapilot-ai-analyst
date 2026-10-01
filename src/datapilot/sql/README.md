# SQL Layer

The `sql` package contains the components responsible for generating, validating, repairing, and executing SQL within DataPilot.

This layer translates natural-language analytical questions into executable database queries and provides the execution boundary between the agent and the underlying data source.

## Responsibilities

The SQL layer is responsible for:

- Generating SQL from natural-language questions.
- Using available schema context during SQL generation.
- Validating generated SQL before execution.
- Executing valid SQL against the configured data source.
- Detecting execution or validation failures.
- Supporting SQL repair and retry workflows.
- Returning structured query results to the agent pipeline.

## Role in DataPilot

The SQL layer sits between the agent orchestration layer and the data source.

```text
User Question
      │
      ▼
 Agent Pipeline
      │
      ▼
 SQL Generator
      │
      ▼
 SQL Validator
      │
      ├── Invalid ──► Repair / Retry
      │
      ▼
 SQL Executor
      │
      ▼
 Query Result
      │
      ▼
 Analytics Layer