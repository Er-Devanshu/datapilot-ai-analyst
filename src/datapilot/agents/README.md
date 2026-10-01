# DataPilot Agents

The `agents` package contains the orchestration layer of DataPilot.

It connects schema inspection, SQL generation, SQL validation, SQL repair, SQL execution, deterministic analysis, visualization selection, evidence construction, and final answer composition into a single agent workflow.

The orchestration is implemented using **LangGraph**.

---

## Package Responsibilities

The `agents` package is responsible for:

1. Maintaining the agent state.
2. Executing the DataPilot workflow in a deterministic sequence.
3. Routing invalid SQL through the repair loop.
4. Preventing unvalidated SQL from reaching execution.
5. Producing visualization specifications from query results.
6. Building grounded evidence from deterministic analysis.
7. Composing the final business answer.
8. Exposing a public runtime interface through `DataPilotAgent`.

The package is intentionally focused on **orchestration**, while individual analytical responsibilities remain in their respective packages.

---

## Package Structure

```text
agents/
├── graph.py
├── nodes.py
├── runtime.py
├── state.py
└── README.md