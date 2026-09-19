from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Contribution:
    """Contribution of one category to an overall change."""

    category: str
    value: float
    contribution_percentage: float | None


@dataclass(frozen=True)
class ContributionResult:
    """Deterministic decomposition of an overall analytical change."""

    dimension_column: str
    value_column: str
    total_change: float
    contributions: tuple[Contribution, ...]


class ContributionAnalyzer:
    """Decompose an overall change across a categorical dimension."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        dimension_column: str,
        value_column: str,
    ) -> ContributionResult:
        """Calculate each category's contribution to the total change."""

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
                "for contribution analysis."
            )

        grouped = (
            working
            .groupby(
                dimension_column,
                dropna=False,
            )[value_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        total_change = float(
            grouped.sum()
        )

        if total_change == 0:
            contributions = tuple(
                Contribution(
                    category=str(category),
                    value=float(value),
                    contribution_percentage=None,
                )
                for category, value
                in grouped.items()
            )
        else:
            contributions = tuple(
                Contribution(
                    category=str(category),
                    value=float(value),
                    contribution_percentage=(
                        float(value)
                        / abs(total_change)
                    )
                    * 100,
                )
                for category, value
                in grouped.items()
            )

        return ContributionResult(
            dimension_column=dimension_column,
            value_column=value_column,
            total_change=total_change,
            contributions=contributions,
        )