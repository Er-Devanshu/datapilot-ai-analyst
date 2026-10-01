# Evaluation Layer

The `evaluation` package contains components used to evaluate DataPilot behavior and validate the quality of its agent pipeline.

The evaluation layer is intended to provide a structured way to test whether DataPilot produces reliable results across its core capabilities.

## Responsibilities

The evaluation layer is responsible for:

- Evaluating agent behavior.
- Supporting repeatable quality checks.
- Measuring expected behavior against actual results.
- Providing a dedicated location for evaluation-related functionality.
- Keeping evaluation concerns separate from the production agent pipeline.

## Role in DataPilot

Evaluation sits outside the core execution path of the agent.

```text
                    ┌─────────────────────┐
                    │      DataPilot      │
                    │    Agent Pipeline   │
                    └──────────┬──────────┘
                               │
                               ▼
                         Agent Results
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Evaluation      │
                    │       Layer         │
                    └──────────┬──────────┘
                               │
                               ▼
                       Quality Assessment