from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.visualization.models import (
    VisualizationSpec,
)


@dataclass(frozen=True)
class VisualizationSelection:
    """Result of deterministic visualization selection."""

    specification: VisualizationSpec
    reason: str


class VisualizationSelector:
    """Select an appropriate visualization from a query result."""

    MAX_CATEGORIES = 20

    def select(
        self,
        dataframe: pd.DataFrame,
        title: str | None = None,
    ) -> VisualizationSelection:
        """Select a visualization based on DataFrame structure."""

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if dataframe.empty:
            return VisualizationSelection(
                specification=VisualizationSpec(
                    chart_type="table",
                    title=title or "No data",
                ),
                reason="The result contains no rows.",
            )

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        datetime_columns = dataframe.select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
        ).columns.tolist()

        temporal_column = self._find_temporal_column(
            dataframe,
            datetime_columns,
            categorical_columns,
        )

        if temporal_column and numeric_columns:
            value_column = numeric_columns[0]

            return VisualizationSelection(
                specification=VisualizationSpec(
                    chart_type="line",
                    x_column=temporal_column,
                    y_column=value_column,
                    title=title
                    or f"{value_column} over {temporal_column}",
                ),
                reason=(
                    "A temporal dimension and numeric measure "
                    "were detected, so a line chart represents "
                    "the trend."
                ),
            )

        if len(numeric_columns) >= 2:
            return VisualizationSelection(
                specification=VisualizationSpec(
                    chart_type="scatter",
                    x_column=numeric_columns[0],
                    y_column=numeric_columns[1],
                    title=title
                    or (
                        f"{numeric_columns[1]} vs "
                        f"{numeric_columns[0]}"
                    ),
                ),
                reason=(
                    "Two numeric measures were detected, "
                    "so a scatter plot can show their relationship."
                ),
            )

        if categorical_columns and numeric_columns:
            dimension = categorical_columns[0]
            measure = numeric_columns[0]

            unique_count = dataframe[dimension].nunique(
                dropna=True
            )

            if unique_count <= self.MAX_CATEGORIES:
                return VisualizationSelection(
                    specification=VisualizationSpec(
                        chart_type="bar",
                        x_column=dimension,
                        y_column=measure,
                        color_column=dimension,
                        title=title
                        or f"{measure} by {dimension}",
                        sort_by=measure,
                        sort_descending=True,
                    ),
                    reason=(
                        "A categorical dimension and numeric "
                        "measure were detected, so a bar chart "
                        "is appropriate."
                    ),
                )

        if numeric_columns:
            measure = numeric_columns[0]

            return VisualizationSelection(
                specification=VisualizationSpec(
                    chart_type="histogram",
                    x_column=measure,
                    title=title
                    or f"Distribution of {measure}",
                ),
                reason=(
                    "Only a numeric measure was detected, "
                    "so a histogram shows its distribution."
                ),
            )

        return VisualizationSelection(
            specification=VisualizationSpec(
                chart_type="table",
                title=title or "Data Result",
            ),
            reason=(
                "No suitable numeric or categorical structure "
                "was detected for a chart."
            ),
        )

    @staticmethod
    def _find_temporal_column(
        dataframe: pd.DataFrame,
        datetime_columns: list[str],
        categorical_columns: list[str],
    ) -> str | None:
        """Identify a temporal column."""

        if datetime_columns:
            return datetime_columns[0]

        temporal_names = {
            "date",
            "datetime",
            "timestamp",
            "month",
            "month_name",
            "quarter",
            "quarter_name",
            "year",
            "week",
        }

        for column in categorical_columns:
            normalized = str(column).strip().lower()

            if normalized in temporal_names:
                return column

        return None