from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ChartType = Literal[
    "bar",
    "line",
    "scatter",
    "histogram",
    "table",
]


@dataclass(frozen=True)
class VisualizationSpec:
    """Deterministic specification for rendering a visualization."""

    chart_type: ChartType
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    title: str | None = None
    orientation: Literal["v", "h"] = "v"
    sort_by: str | None = None
    sort_descending: bool = True
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate the visualization specification."""

        if self.chart_type not in {
            "bar",
            "line",
            "scatter",
            "histogram",
            "table",
        }:
            raise ValueError(
                f"Unsupported chart type: {self.chart_type}"
            )

        if self.limit is not None and self.limit <= 0:
            raise ValueError(
                "Visualization limit must be greater than zero."
            )

        if self.orientation not in {"v", "h"}:
            raise ValueError(
                "Visualization orientation must be 'v' or 'h'."
            )

    @property
    def requires_x(self) -> bool:
        """Return whether the chart requires an x-axis column."""

        return self.chart_type in {
            "bar",
            "line",
            "scatter",
        }

    @property
    def requires_y(self) -> bool:
        """Return whether the chart requires a y-axis column."""

        return self.chart_type in {
            "bar",
            "line",
            "scatter",
        }