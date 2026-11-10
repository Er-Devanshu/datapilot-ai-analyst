
from __future__ import annotations

import calendar
import re
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
        """Analyze a time-series result in deterministic period order."""

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
        ].dropna().copy()

        if working.empty:
            raise ValueError(
                "No valid observations available for trend analysis."
            )

        working = self._sort_periods(
            working,
            period_column,
        )

        # Use column arrays rather than iterrows(). iterrows() can coerce
        # integer period values to floats when another column is floating.
        periods = working[period_column].tolist()
        values = working[value_column].tolist()

        points = tuple(
            TrendPoint(
                period=str(period),
                value=float(value),
            )
            for period, value in zip(
                periods,
                values,
                strict=True,
            )
        )

        first_value = points[0].value
        last_value = points[-1].value
        absolute_change = last_value - first_value

        if first_value == 0:
            percentage_change = None
        else:
            percentage_change = (
                absolute_change / abs(first_value)
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

    @staticmethod
    def _sort_periods(
        dataframe: pd.DataFrame,
        period_column: str,
    ) -> pd.DataFrame:
        """Sort periods using the strongest unambiguous ordering available."""

        periods = dataframe[period_column]

        if pd.api.types.is_datetime64_any_dtype(periods):
            return dataframe.sort_values(
                by=period_column,
                kind="mergesort",
            )

        if pd.api.types.is_numeric_dtype(periods):
            return dataframe.sort_values(
                by=period_column,
                kind="mergesort",
            )

        normalized = periods.astype(str).str.strip()

        # ISO-style year and month/date labels.
        iso_pattern = re.compile(
            r"\d{4}-\d{2}(?:-\d{2})?"
        )

        if normalized.map(
            lambda value: bool(
                iso_pattern.fullmatch(value)
            )
        ).all():
            working = dataframe.copy()
            working["_datapilot_period_sort"] = (
                pd.to_datetime(
                    normalized,
                    format="mixed",
                    errors="raise",
                )
            )

            working = working.sort_values(
                by="_datapilot_period_sort",
                kind="mergesort",
            )

            return working.drop(
                columns=["_datapilot_period_sort"]
            )

        # Month names without a year have an unambiguous calendar order.
        month_lookup = {
            name.casefold(): month
            for month in range(1, 13)
            for name in (
                calendar.month_name[month],
                calendar.month_abbr[month],
            )
        }

        month_keys = normalized.str.casefold()

        if month_keys.isin(month_lookup).all():
            working = dataframe.copy()
            working["_datapilot_period_sort"] = (
                month_keys.map(month_lookup)
            )

            working = working.sort_values(
                by="_datapilot_period_sort",
                kind="mergesort",
            )

            return working.drop(
                columns=["_datapilot_period_sort"]
            )

        # Other labels have no guaranteed chronological interpretation.
        return dataframe.sort_values(
            by=period_column,
            kind="mergesort",
        )
