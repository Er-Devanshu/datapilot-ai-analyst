from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from datapilot.visualization.models import (
    VisualizationSpec,
)


@dataclass(frozen=True)
class RenderedVisualization:
    """Rendered visualization and its specification."""

    specification: VisualizationSpec
    figure: go.Figure


class VisualizationRenderer:
    """Render visualization specifications using Plotly."""

    def render(
        self,
        dataframe: pd.DataFrame,
        specification: VisualizationSpec,
    ) -> RenderedVisualization:
        """Render a DataFrame according to a visualization spec."""

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if specification is None:
            raise ValueError(
                "Visualization specification cannot be None."
            )

        self._validate_columns(
            dataframe,
            specification,
        )

        working = dataframe.copy()

        if specification.limit is not None:
            working = self._apply_limit(
                working,
                specification,
            )

        if specification.chart_type == "bar":
            figure = px.bar(
                working,
                x=specification.x_column,
                y=specification.y_column,
                color=specification.color_column,
                title=specification.title,
                orientation=specification.orientation,
            )

        elif specification.chart_type == "line":
            figure = px.line(
                working,
                x=specification.x_column,
                y=specification.y_column,
                color=specification.color_column,
                title=specification.title,
                markers=True,
            )

        elif specification.chart_type == "scatter":
            figure = px.scatter(
                working,
                x=specification.x_column,
                y=specification.y_column,
                color=specification.color_column,
                title=specification.title,
            )

        elif specification.chart_type == "histogram":
            figure = px.histogram(
                working,
                x=specification.x_column,
                title=specification.title,
            )

        elif specification.chart_type == "table":
            figure = self._render_table(
                working,
                specification,
            )

        else:
            raise ValueError(
                f"Unsupported chart type: "
                f"{specification.chart_type}"
            )

        return RenderedVisualization(
            specification=specification,
            figure=figure,
        )

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        specification: VisualizationSpec,
    ) -> None:
        """Validate all columns referenced by the specification."""

        available = {
            str(column)
            for column in dataframe.columns
        }

        referenced = {
            column
            for column in {
                specification.x_column,
                specification.y_column,
                specification.color_column,
                specification.sort_by,
            }
            if column is not None
        }

        missing = referenced - available

        if missing:
            raise ValueError(
                "Visualization references unknown columns: "
                + ", ".join(sorted(missing))
            )

        if specification.requires_x and not specification.x_column:
            raise ValueError(
                "This chart type requires an x-axis column."
            )

        if specification.requires_y and not specification.y_column:
            raise ValueError(
                "This chart type requires a y-axis column."
            )

    @staticmethod
    def _apply_limit(
        dataframe: pd.DataFrame,
        specification: VisualizationSpec,
    ) -> pd.DataFrame:
        """Apply deterministic sorting and row limiting."""

        working = dataframe

        if specification.sort_by is not None:
            working = working.sort_values(
                by=specification.sort_by,
                ascending=not specification.sort_descending,
            )

        return working.head(
            specification.limit
        )

    @staticmethod
    def _render_table(
        dataframe: pd.DataFrame,
        specification: VisualizationSpec,
    ) -> go.Figure:
        """Render a tabular result as a Plotly table."""

        return go.Figure(
            data=[
                go.Table(
                    header={
                        "values": list(
                            dataframe.columns
                        )
                    },
                    cells={
                        "values": [
                            dataframe[column].tolist()
                            for column in dataframe.columns
                        ]
                    },
                )
            ],
            layout={
                "title": specification.title,
            },
        )