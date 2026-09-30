from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SegmentSummary:
    """Deterministic summary of one analytical segment."""

    segment: str
    row_count: int
    total_value: float
    average_value: float
    minimum_value: float
    maximum_value: float


@dataclass(frozen=True)
class SegmentationResult:
    """Deterministic analysis of a categorical segmentation."""

    dimension_column: str
    value_column: str
    total_value: float
    segments: tuple[SegmentSummary, ...]


class SegmentationAnalyzer:
    """Profile and compare segments deterministically."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        dimension_column: str,
        value_column: str,
    ) -> SegmentationResult:
        """Calculate descriptive statistics for each segment."""

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if dataframe.empty:
            raise ValueError(
                "DataFrame cannot be empty."
            )

        if dimension_column not in dataframe.columns:
            raise ValueError(
                f"Dimension column does not exist: "
                f"{dimension_column}"
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
            [dimension_column, value_column]
        ].dropna()

        if working.empty:
            raise ValueError(
                "No valid observations available "
                "for segmentation analysis."
            )

        grouped = (
            working
            .groupby(
                dimension_column,
                dropna=False,
            )[value_column]
            .agg(
                [
                    "count",
                    "sum",
                    "mean",
                    "min",
                    "max",
                ]
            )
            .sort_values(
                "sum",
                ascending=False,
            )
        )

        segments = tuple(
            SegmentSummary(
                segment=str(segment),
                row_count=int(row["count"]),
                total_value=float(row["sum"]),
                average_value=float(row["mean"]),
                minimum_value=float(row["min"]),
                maximum_value=float(row["max"]),
            )
            for segment, row in grouped.iterrows()
        )

        total_value = float(
            working[value_column].sum()
        )

        return SegmentationResult(
            dimension_column=dimension_column,
            value_column=value_column,
            total_value=total_value,
            segments=segments,
        )