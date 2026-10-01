# Visualization

The `visualization` package is responsible for selecting and representing visualizations for DataPilot query results.

It sits after analysis and evidence generation in the agent pipeline and converts a tabular result into a deterministic visualization specification that can be rendered by downstream consumers.

## Responsibilities

The package handles:

- Visualization specification models
- Chart-type selection
- Dimension and measure detection
- Deterministic visualization decisions
- Visualization metadata
- Rendering support for downstream interfaces

The package does **not** perform SQL generation or data analysis. It consumes the structured result produced by earlier stages of the DataPilot pipeline.

## Package Structure

```text
visualization/
├── __init__.py
├── models.py
├── selector.py
├── renderer.py
└── README.md