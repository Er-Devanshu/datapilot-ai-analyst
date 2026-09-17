from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NumericSummary:
    """Summary statistics for a numeric result column."""

    column: str
    total: float
    average: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class CategorySummary:
    """Summary of a categorical dimension and numeric measure."""

    dimension: str
    measure: str
    top_category: str
    top_value: float
    bottom_category: str
    bottom_value: float


@dataclass(frozen=True)
class AnalysisResult:
    """Deterministic analytical summary of a query result."""

    row_count: int
    numeric_summaries: tuple[NumericSummary, ...]
    category_summaries: tuple[CategorySummary, ...]


class ResultAnalyzer:
    """Perform deterministic analysis of query results."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> AnalysisResult:
        """Analyze a DataFrame without using an LLM."""

        if dataframe is None:
            raise ValueError(
                "DataFrame cannot be None."
            )

        if dataframe.empty:
            return AnalysisResult(
                row_count=0,
                numeric_summaries=(),
                category_summaries=(),
            )

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        numeric_summaries = tuple(
            self._summarize_numeric(
                dataframe,
                column,
            )
            for column in numeric_columns
        )

        category_summaries: list[CategorySummary] = []

        for dimension in categorical_columns:
            for measure in numeric_columns:
                summary = self._summarize_category(
                    dataframe,
                    dimension,
                    measure,
                )

                if summary is not None:
                    category_summaries.append(
                        summary
                    )

        return AnalysisResult(
            row_count=len(dataframe),
            numeric_summaries=numeric_summaries,
            category_summaries=tuple(
                category_summaries
            ),
        )

    @staticmethod
    def _summarize_numeric(
        dataframe: pd.DataFrame,
        column: str,
    ) -> NumericSummary:
        """Calculate summary statistics for a numeric column."""

        series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            return NumericSummary(
                column=column,
                total=0.0,
                average=0.0,
                minimum=0.0,
                maximum=0.0,
            )

        return NumericSummary(
            column=column,
            total=float(series.sum()),
            average=float(series.mean()),
            minimum=float(series.min()),
            maximum=float(series.max()),
        )

    @staticmethod
    def _summarize_category(
        dataframe: pd.DataFrame,
        dimension: str,
        measure: str,
    ) -> CategorySummary | None:
        """Find the highest and lowest categories for a measure."""

        working = dataframe[
            [dimension, measure]
        ].copy()

        working[measure] = pd.to_numeric(
            working[measure],
            errors="coerce",
        )

        working = working.dropna(
            subset=[dimension, measure]
        )

        if working.empty:
            return None

        grouped = (
            working
            .groupby(dimension, dropna=False)[measure]
            .sum()
            .sort_values()
        )

        if grouped.empty:
            return None

        bottom_category = str(
            grouped.index[0]
        )

        top_category = str(
            grouped.index[-1]
        )

        return CategorySummary(
            dimension=dimension,
            measure=measure,
            top_category=top_category,
            top_value=float(
                grouped.iloc[-1]
            ),
            bottom_category=bottom_category,
            bottom_value=float(
                grouped.iloc[0]
            ),
        )