from __future__ import annotations

import pandas as pd

from datapilot.analytics.analyzer import ResultAnalyzer


def test_empty_dataframe() -> None:
    result = ResultAnalyzer().analyze(
        pd.DataFrame()
    )

    assert result.row_count == 0
    assert result.numeric_summaries == ()
    assert result.category_summaries == ()


def test_numeric_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    result = ResultAnalyzer().analyze(
        dataframe
    )

    assert result.row_count == 3
    assert len(result.numeric_summaries) == 1

    summary = result.numeric_summaries[0]

    assert summary.column == "revenue"
    assert summary.total == 600.0
    assert summary.average == 200.0
    assert summary.minimum == 100.0
    assert summary.maximum == 300.0


def test_category_summary() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "West",
                "West",
            ],
            "revenue": [
                1000.0,
                2000.0,
                5000.0,
                1000.0,
            ],
        }
    )

    result = ResultAnalyzer().analyze(
        dataframe
    )

    summaries = [
        summary
        for summary in result.category_summaries
        if (
            summary.dimension == "region"
            and summary.measure == "revenue"
        )
    ]

    assert len(summaries) == 1

    summary = summaries[0]

    assert summary.top_category == "West"
    assert summary.top_value == 6000.0

    assert summary.bottom_category == "North"
    assert summary.bottom_value == 3000.0


def test_missing_numeric_values_are_ignored() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "West",
            ],
            "revenue": [
                1000.0,
                None,
                3000.0,
            ],
        }
    )

    result = ResultAnalyzer().analyze(
        dataframe
    )

    summary = [
        item
        for item in result.category_summaries
        if (
            item.dimension == "region"
            and item.measure == "revenue"
        )
    ][0]

    assert summary.top_category == "West"
    assert summary.top_value == 3000.0
    assert summary.bottom_category == "North"
    assert summary.bottom_value == 1000.0