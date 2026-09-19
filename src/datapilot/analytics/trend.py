from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrendPoint:
    """A single point in a time-ordered analytical trend."""

    period: str
    value: float


@dataclass(frozen=True)
class TrendResult:
    """Deterministic summary of a time-series trend."""

    period_column: str
    value_column: str
    points: tuple[TrendPoint, ...]
    first_value: float
    last_value: float
    absolute_change: float
    percentage_change: float | None
    direction: str


class TrendAnalyzer:
    """Perform deterministic trend analysis on query results."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        period_column: str,
        value_column: str,
    ) -> TrendResult:
        """Analyze an ordered time-series result."""

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if dataframe.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        if period_column not in dataframe.columns:
            raise ValueError(
                f"Period column does not exist: "
                f"{period_column}"
            )

        if value_column not in dataframe.columns:
            raise ValueError(
                f"Value column does not exist: "
                f"{value_column}"
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[value_column]
        ):
            raise ValueError(
                f"Value column must be numeric: "
                f"{value_column}"
            )

        working = dataframe[
            [period_column, value_column]
        ].dropna()

        if working.empty:
            raise ValueError(
                "No valid observations available for trend analysis."
            )

        points = tuple(
            TrendPoint(
                period=str(row[period_column]),
                value=float(row[value_column]),
            )
            for _, row in working.iterrows()
        )

        first_value = points[0].value
        last_value = points[-1].value
        absolute_change = (
            last_value - first_value
        )

        if first_value == 0:
            percentage_change = None
        else:
            percentage_change = (
                absolute_change
                / abs(first_value)
            ) * 100

        if absolute_change > 0:
            direction = "increasing"
        elif absolute_change < 0:
            direction = "decreasing"
        else:
            direction = "stable"

        return TrendResult(
            period_column=period_column,
            value_column=value_column,
            points=points,
            first_value=first_value,
            last_value=last_value,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            direction=direction,
        )