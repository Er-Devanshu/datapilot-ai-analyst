from __future__ import annotations

import pytest

from datapilot.analytics.contribution import (
    Contribution,
    ContributionResult,
)
from datapilot.analytics.root_cause import (
    RootCauseAnalyzer,
)
from datapilot.analytics.variance import (
    VarianceResult,
)


def test_root_cause_analysis() -> None:
    variance = VarianceResult(
        actual_value=85,
        reference_value=100,
        absolute_variance=-15,
        percentage_variance=-15,
        direction="below_reference",
    )

    contribution = ContributionResult(
        dimension_column="region",
        value_column="revenue_change",
        total_change=-15,
        contributions=(
            Contribution(
                category="North",
                value=-8,
                contribution_percentage=-53.3333333333,
            ),
            Contribution(
                category="West",
                value=-5,
                contribution_percentage=-33.3333333333,
            ),
            Contribution(
                category="South",
                value=-3,
                contribution_percentage=-20,
            ),
            Contribution(
                category="East",
                value=1,
                contribution_percentage=6.6666666667,
            ),
        ),
    )

    result = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    assert result.dimension_column == "region"
    assert result.value_column == "revenue_change"
    assert result.overall_change == -15
    assert result.overall_direction == "below_reference"

    assert len(result.drivers) == 4

    assert result.drivers[0].category == "North"
    assert result.drivers[0].value == -8
    assert result.drivers[0].direction == "negative"

    assert result.drivers[1].category == "West"
    assert result.drivers[1].direction == "negative"

    assert result.drivers[2].category == "South"
    assert result.drivers[2].direction == "negative"

    assert result.drivers[3].category == "East"
    assert result.drivers[3].value == 1
    assert result.drivers[3].direction == "positive"


def test_root_cause_accepts_zero_change() -> None:
    variance = VarianceResult(
        actual_value=100,
        reference_value=100,
        absolute_variance=0,
        percentage_variance=0,
        direction="at_reference",
    )

    contribution = ContributionResult(
        dimension_column="region",
        value_column="revenue_change",
        total_change=0,
        contributions=(
            Contribution(
                category="North",
                value=0,
                contribution_percentage=None,
            ),
        ),
    )

    result = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    assert result.overall_change == 0
    assert result.overall_direction == "at_reference"
    assert result.drivers[0].direction == "neutral"


def test_root_cause_rejects_mismatched_totals() -> None:
    variance = VarianceResult(
        actual_value=85,
        reference_value=100,
        absolute_variance=-15,
        percentage_variance=-15,
        direction="below_reference",
    )

    contribution = ContributionResult(
        dimension_column="region",
        value_column="revenue_change",
        total_change=-10,
        contributions=(
            Contribution(
                category="North",
                value=-10,
                contribution_percentage=-100,
            ),
        ),
    )

    with pytest.raises(ValueError):
        RootCauseAnalyzer().analyze(
            variance=variance,
            contribution=contribution,
        )


def test_root_cause_requires_variance() -> None:
    contribution = ContributionResult(
        dimension_column="region",
        value_column="revenue_change",
        total_change=-10,
        contributions=(),
    )

    with pytest.raises(ValueError):
        RootCauseAnalyzer().analyze(
            variance=None,
            contribution=contribution,
        )


def test_root_cause_requires_contribution() -> None:
    variance = VarianceResult(
        actual_value=90,
        reference_value=100,
        absolute_variance=-10,
        percentage_variance=-10,
        direction="below_reference",
    )

    with pytest.raises(ValueError):
        RootCauseAnalyzer().analyze(
            variance=variance,
            contribution=None,
        )