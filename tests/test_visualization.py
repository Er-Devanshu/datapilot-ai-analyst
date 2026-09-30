from __future__ import annotations

import pandas as pd
import pytest

from datapilot.visualization.models import (
    VisualizationSpec,
)
from datapilot.visualization.renderer import (
    VisualizationRenderer,
)
from datapilot.visualization.selector import (
    VisualizationSelector,
)


def test_selector_chooses_bar_for_category_and_measure() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "West",
                "East",
                "North",
            ],
            "revenue": [
                100,
                80,
                90,
            ],
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "bar"
    assert result.specification.x_column == "region"
    assert result.specification.y_column == "revenue"


def test_selector_chooses_line_for_temporal_result() -> None:
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-03-01",
                ]
            ),
            "revenue": [
                100,
                120,
                140,
            ],
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "line"
    assert result.specification.x_column == "date"
    assert result.specification.y_column == "revenue"


def test_selector_recognizes_month_as_temporal() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "January",
                "February",
                "March",
            ],
            "revenue": [
                100,
                120,
                140,
            ],
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "line"
    assert result.specification.x_column == "month"


def test_selector_chooses_scatter_for_two_numeric_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "discount": [10, 20, 30],
            "profit": [40, 50, 70],
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "scatter"
    assert result.specification.x_column == "discount"
    assert result.specification.y_column == "profit"


def test_selector_chooses_histogram_for_single_numeric_column() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
            ]
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "histogram"
    assert result.specification.x_column == "revenue"


def test_selector_falls_back_to_table() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "West",
                "East",
            ],
            "category": [
                "Technology",
                "Furniture",
            ],
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "table"


def test_selector_returns_table_for_empty_result() -> None:
    dataframe = pd.DataFrame(
        {
            "region": pd.Series(dtype="string"),
            "revenue": pd.Series(dtype="float64"),
        }
    )

    result = VisualizationSelector().select(
        dataframe
    )

    assert result.specification.chart_type == "table"


def test_renderer_creates_bar_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "West",
                "East",
            ],
            "revenue": [
                100,
                80,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="revenue",
        title="Revenue by Region",
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    assert result.specification is specification
    assert len(result.figure.data) == 1
    assert result.figure.data[0].type == "bar"


def test_renderer_creates_line_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "month": [
                "January",
                "February",
            ],
            "revenue": [
                100,
                120,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="line",
        x_column="month",
        y_column="revenue",
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    assert result.figure.data[0].type == "scatter"


def test_renderer_creates_scatter_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "discount": [10, 20],
            "profit": [30, 50],
        }
    )

    specification = VisualizationSpec(
        chart_type="scatter",
        x_column="discount",
        y_column="profit",
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    assert result.figure.data[0].type == "scatter"


def test_renderer_creates_histogram() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
            ]
        }
    )

    specification = VisualizationSpec(
        chart_type="histogram",
        x_column="revenue",
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    assert result.figure.data[0].type == "histogram"


def test_renderer_creates_table() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "West",
                "East",
            ],
            "revenue": [
                100,
                80,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="table",
        title="Revenue",
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    assert result.figure.data[0].type == "table"


def test_renderer_rejects_unknown_column() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "West",
            ],
            "revenue": [
                100,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="profit",
    )

    with pytest.raises(ValueError):
        VisualizationRenderer().render(
            dataframe,
            specification,
        )


def test_renderer_applies_sort_and_limit() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "East",
                "West",
                "North",
            ],
            "revenue": [
                100,
                300,
                200,
            ],
        }
    )

    specification = VisualizationSpec(
        chart_type="bar",
        x_column="region",
        y_column="revenue",
        sort_by="revenue",
        sort_descending=True,
        limit=2,
    )

    result = VisualizationRenderer().render(
        dataframe,
        specification,
    )

    x_values = list(
        result.figure.data[0].x
    )

    assert x_values == [
        "West",
        "North",
    ]


def test_visualization_spec_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError):
        VisualizationSpec(
            chart_type="bar",
            limit=0,
        )


def test_visualization_spec_rejects_invalid_orientation() -> None:
    with pytest.raises(ValueError):
        VisualizationSpec(
            chart_type="bar",
            orientation="x",
        )