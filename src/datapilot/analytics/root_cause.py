from __future__ import annotations

from dataclasses import dataclass

from datapilot.analytics.contribution import ContributionResult
from datapilot.analytics.variance import VarianceResult


@dataclass(frozen=True)
class RootCauseDriver:
    """An evidence-backed driver of an overall analytical change."""

    category: str
    value: float
    contribution_percentage: float | None
    direction: str


@dataclass(frozen=True)
class RootCauseResult:
    """Deterministic root-cause analysis result."""

    dimension_column: str
    value_column: str
    overall_change: float
    overall_direction: str
    drivers: tuple[RootCauseDriver, ...]


class RootCauseAnalyzer:
    """Identify evidence-backed drivers of an overall change."""

    def analyze(
        self,
        variance: VarianceResult,
        contribution: ContributionResult,
    ) -> RootCauseResult:
        """Combine overall variance and contribution evidence."""

        if variance is None:
            raise ValueError(
                "Variance result cannot be None."
            )

        if contribution is None:
            raise ValueError(
                "Contribution result cannot be None."
            )

        if (
            abs(
                variance.absolute_variance
                - contribution.total_change
            )
            > 1e-9
        ):
            raise ValueError(
                "Variance and contribution totals do not match."
            )

        drivers = tuple(
            RootCauseDriver(
                category=item.category,
                value=item.value,
                contribution_percentage=(
                    item.contribution_percentage
                ),
                direction=(
                    "positive"
                    if item.value > 0
                    else "negative"
                    if item.value < 0
                    else "neutral"
                ),
            )
            for item in contribution.contributions
        )

        return RootCauseResult(
            dimension_column=contribution.dimension_column,
            value_column=contribution.value_column,
            overall_change=variance.absolute_variance,
            overall_direction=variance.direction,
            drivers=drivers,
        )