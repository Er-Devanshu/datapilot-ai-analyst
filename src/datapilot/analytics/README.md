# DataPilot Analytics

The `analytics` package contains the deterministic analytical and answer-generation layer of DataPilot.

It is responsible for transforming executed query results into:

1. Structured analytical insights.
2. Grounded evidence.
3. A business-friendly final answer.

The package deliberately separates **deterministic computation** from **LLM-based language generation**.

---

## Package Structure

```text
analytics/
├── analyzer.py
├── answer.py
├── evidence.py
└── README.md