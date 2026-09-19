from __future__ import annotations

import pandas as pd
import pytest

from datapilot.analytics.contribution import (
    ContributionAnalyzer,
)


def test_contribution_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
                "South",
                "East",
            ],
            "revenue_change": [
                -8_000_000,
                -5_000_000,
                -3_000_000,
                1_000_000,
            ],
        }
    )

    result = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    assert result.dimension_column == "region"
    assert result.value_column == "revenue_change"
    assert result.total_change == -15_000_000

    assert result.contributions[0].category == "East"
    assert result.contributions[0].value == 1_000_000

    assert result.contributions[1].category == "South"
    assert result.contributions[1].value == -3_000_000

    assert result.contributions[2].category == "West"
    assert result.contributions[2].value == -5_000_000

    assert result.contributions[3].category == "North"
    assert result.contributions[3].value == -8_000_000


def test_contribution_percentages() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
            ],
            "revenue_change": [
                -60,
                -40,
            ],
        }
    )

    result = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    contributions = {
        item.category: item
        for item in result.contributions
    }

    assert contributions["North"].contribution_percentage == -60
    assert contributions["West"].contribution_percentage == -40


def test_zero_total_change() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
            ],
            "revenue_change": [
                100,
                -100,
            ],
        }
    )

    result = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    assert result.total_change == 0

    for contribution in result.contributions:
        assert contribution.contribution_percentage is None


def test_missing_dimension_column() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue_change": [100, 200],
        }
    )

    with pytest.raises(ValueError):
        ContributionAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="region",
            value_column="revenue_change",
        )


def test_missing_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["North", "West"],
        }
    )

    with pytest.raises(ValueError):
        ContributionAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="region",
            value_column="revenue_change",
        )


def test_non_numeric_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["North", "West"],
            "revenue_change": ["100", "200"],
        }
    )

    with pytest.raises(ValueError):
        ContributionAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="region",
            value_column="revenue_change",
        )


def test_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "region",
            "revenue_change",
        ]
    )

    with pytest.raises(ValueError):
        ContributionAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="region",
            value_column="revenue_change",
        )