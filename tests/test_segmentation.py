from __future__ import annotations

import pandas as pd
import pytest

from datapilot.analytics.segmentation import (
    SegmentationAnalyzer,
)


def test_segmentation_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_segment": [
                "Enterprise",
                "Enterprise",
                "SMB",
                "SMB",
                "Consumer",
            ],
            "revenue": [
                500,
                300,
                200,
                100,
                50,
            ],
        }
    )

    result = SegmentationAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="customer_segment",
        value_column="revenue",
    )

    assert result.dimension_column == "customer_segment"
    assert result.value_column == "revenue"
    assert result.total_value == 1150

    assert len(result.segments) == 3

    enterprise = result.segments[0]

    assert enterprise.segment == "Enterprise"
    assert enterprise.row_count == 2
    assert enterprise.total_value == 800
    assert enterprise.average_value == 400
    assert enterprise.minimum_value == 300
    assert enterprise.maximum_value == 500


def test_segments_are_sorted_by_total_value() -> None:
    dataframe = pd.DataFrame(
        {
            "segment": [
                "A",
                "B",
                "C",
            ],
            "revenue": [
                100,
                500,
                250,
            ],
        }
    )

    result = SegmentationAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="segment",
        value_column="revenue",
    )

    assert [
        segment.segment
        for segment in result.segments
    ] == [
        "B",
        "C",
        "A",
    ]


def test_missing_dimension_column() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200],
        }
    )

    with pytest.raises(ValueError):
        SegmentationAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="segment",
            value_column="revenue",
        )


def test_missing_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B"],
        }
    )

    with pytest.raises(ValueError):
        SegmentationAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="segment",
            value_column="revenue",
        )


def test_non_numeric_value_column() -> None:
    dataframe = pd.DataFrame(
        {
            "segment": ["A", "B"],
            "revenue": ["100", "200"],
        }
    )

    with pytest.raises(ValueError):
        SegmentationAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="segment",
            value_column="revenue",
        )


def test_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "segment",
            "revenue",
        ]
    )

    with pytest.raises(ValueError):
        SegmentationAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="segment",
            value_column="revenue",
        )


def test_no_valid_observations() -> None:
    dataframe = pd.DataFrame(
        {
            "segment": [None, None],
            "revenue": [100, 200],
        }
    )

    with pytest.raises(ValueError):
        SegmentationAnalyzer().analyze(
            dataframe=dataframe,
            dimension_column="segment",
            value_column="revenue",
        )