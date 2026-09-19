import pandas as pd
import pytest

from datapilot.analytics.trend import TrendAnalyzer


def test_trend_analysis_detects_increasing_trend() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            "revenue": [
                100.0,
                125.0,
                150.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert result.period_column == "month"
    assert result.value_column == "revenue"

    assert result.first_value == 100.0
    assert result.last_value == 150.0

    assert result.absolute_change == 50.0
    assert result.percentage_change == 50.0

    assert result.direction == "increasing"

    assert len(result.points) == 3


def test_trend_analysis_detects_decreasing_trend() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            "revenue": [
                200.0,
                150.0,
                100.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert result.absolute_change == -100.0
    assert result.percentage_change == -50.0
    assert result.direction == "decreasing"


def test_trend_analysis_detects_stable_trend() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            "revenue": [
                100.0,
                100.0,
                100.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert result.absolute_change == 0.0
    assert result.percentage_change == 0.0
    assert result.direction == "stable"


def test_trend_analysis_handles_zero_start_value() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
            ],
            "revenue": [
                0.0,
                100.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert result.absolute_change == 100.0
    assert result.percentage_change is None
    assert result.direction == "increasing"


def test_trend_analysis_preserves_input_order() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-03",
                "2025-01",
                "2025-02",
            ],
            "revenue": [
                300.0,
                100.0,
                200.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert [
        point.period
        for point in result.points
    ] == [
        "2025-03",
        "2025-01",
        "2025-02",
    ]


def test_trend_analysis_rejects_missing_period_column() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Period column does not exist",
    ):
        TrendAnalyzer().analyze(
            dataframe=dataframe,
            period_column="month",
            value_column="revenue",
        )


def test_trend_analysis_rejects_missing_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Value column does not exist",
    ):
        TrendAnalyzer().analyze(
            dataframe=dataframe,
            period_column="month",
            value_column="revenue",
        )


def test_trend_analysis_rejects_non_numeric_value() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
            ],
            "revenue": [
                "100",
                "200",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Value column must be numeric",
    ):
        TrendAnalyzer().analyze(
            dataframe=dataframe,
            period_column="month",
            value_column="revenue",
        )


def test_trend_analysis_rejects_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "month",
            "revenue",
        ]
    )

    with pytest.raises(
        ValueError,
        match="DataFrame cannot be empty",
    ):
        TrendAnalyzer().analyze(
            dataframe=dataframe,
            period_column="month",
            value_column="revenue",
        )


def test_trend_analysis_ignores_missing_observations() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            "revenue": [
                100.0,
                None,
                150.0,
            ],
        }
    )

    result = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    assert len(result.points) == 2
    assert result.first_value == 100.0
    assert result.last_value == 150.0
    assert result.absolute_change == 50.0