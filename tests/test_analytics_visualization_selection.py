
from __future__ import annotations

import json

import pandas as pd

from datapilot.analytics.executor import (
    AnalysisExecution,
    AnalyticsExecutionResult,
    AnalyticsExecutor,
)
from datapilot.analytics.planner import AnalyticsPlanner
from datapilot.analytics.planning import AnalysisType
from datapilot.analytics.trend import TrendResult
from datapilot.visualization.analytics_selection import (
    AnalyticsAwareVisualizationSelector,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def build_execution(
    dataframe: pd.DataFrame,
    response: dict,
) -> AnalyticsExecutionResult:
    planning = AnalyticsPlanner(
        FakeLLM(json.dumps(response))
    ).plan(
        question="Analyze the supplied data.",
        dataframe=dataframe,
    )

    assert planning.validation.fatal_errors == ()
    assert len(planning.validation.accepted_requests) == 1

    return AnalyticsExecutor().execute(
        dataframe=dataframe,
        validation=planning.validation,
    )


def build_empty_trend_execution() -> AnalysisExecution:
    return AnalysisExecution(
        request_id="empty_trend",
        analysis_type=AnalysisType.TREND,
        result=TrendResult(
            period_column="month",
            value_column="revenue",
            points=(),
            first_value=0.0,
            last_value=0.0,
            absolute_change=0.0,
            percentage_change=0.0,
            direction="stable",
        ),
    )


def test_selector_prefers_segmentation_analytics() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "North"],
            "revenue": [700.0, 200.0, 100.0],
        }
    )

    execution = build_execution(
        dataframe,
        {
            "requests": [
                {
                    "request_id": "revenue_by_region",
                    "analysis_type": "segmentation",
                    "inputs": {
                        "dimension_column": "region",
                        "value_column": "revenue",
                    },
                    "depends_on": [],
                }
            ]
        },
    )

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe,
        execution,
    )

    assert selection.source == "analytics"
    assert selection.request_id == "revenue_by_region"
    assert selection.specification.chart_type == "bar"
    assert selection.specification.x_column == "segment"
    assert selection.specification.y_column == "total_value"
    assert selection.dataframe["segment"].tolist() == [
        "West",
        "East",
        "North",
    ]
    assert selection.dataframe["total_value"].tolist() == [
        700.0,
        200.0,
        100.0,
    ]

    # The original SQL result remains unchanged.
    assert dataframe.columns.tolist() == ["region", "revenue"]


def test_selector_uses_analytics_table_for_variance() -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["June", "June", "July", "July"],
            "revenue": [60.0, 40.0, 50.0, 35.0],
        }
    )

    execution = build_execution(
        dataframe,
        {
            "requests": [
                {
                    "request_id": "revenue_variance",
                    "analysis_type": "variance",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "reference_period": "June",
                        "actual_period": "July",
                        "aggregation": "sum",
                    },
                    "depends_on": [],
                }
            ]
        },
    )

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe,
        execution,
    )

    assert selection.source == "analytics"
    assert selection.request_id == "revenue_variance"
    assert selection.specification.chart_type == "table"
    assert selection.specification.x_column is None
    assert selection.specification.y_column is None
    assert selection.dataframe["metric"].tolist() == [
        "Actual value",
        "Reference value",
        "Absolute variance",
        "Percentage variance",
    ]


def test_selector_falls_back_when_no_analytics_execution_exists() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [700.0, 200.0],
        }
    )

    execution = AnalyticsExecutionResult(
        executions=(),
        skipped_analyses=(),
        fatal_errors=(),
    )

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe,
        execution,
    )

    assert selection.source == "query_result"
    assert selection.request_id is None
    assert selection.dataframe is dataframe
    assert selection.specification.x_column == "region"
    assert selection.specification.y_column == "revenue"


def test_selector_falls_back_when_analytics_execution_is_missing() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    selection = AnalyticsAwareVisualizationSelector().select(dataframe)

    assert selection.source == "query_result"
    assert selection.request_id is None
    assert selection.dataframe is dataframe
    assert selection.specification.chart_type == "histogram"


def test_selector_skips_empty_analytics_and_uses_next_supported_result() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [700.0, 200.0],
        }
    )

    valid_execution = build_execution(
        dataframe,
        {
            "requests": [
                {
                    "request_id": "revenue_by_region",
                    "analysis_type": "segmentation",
                    "inputs": {
                        "dimension_column": "region",
                        "value_column": "revenue",
                    },
                    "depends_on": [],
                }
            ]
        },
    )

    execution = AnalyticsExecutionResult(
        executions=(
            build_empty_trend_execution(),
            *valid_execution.executions,
        ),
        skipped_analyses=valid_execution.skipped_analyses,
        fatal_errors=valid_execution.fatal_errors,
    )

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe,
        execution,
    )

    assert selection.source == "analytics"
    assert selection.request_id == "revenue_by_region"
    assert selection.specification.chart_type == "bar"
    assert selection.dataframe["segment"].tolist() == [
        "West",
        "East",
    ]


def test_selector_falls_back_when_all_analytics_results_are_empty() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    execution = AnalyticsExecutionResult(
        executions=(build_empty_trend_execution(),),
        skipped_analyses=(),
        fatal_errors=(),
    )

    selection = AnalyticsAwareVisualizationSelector().select(
        dataframe,
        execution,
    )

    assert selection.source == "query_result"
    assert selection.request_id is None
    assert selection.dataframe is dataframe
    assert selection.specification.chart_type == "histogram"


def test_selector_rejects_non_dataframe_input() -> None:
    try:
        AnalyticsAwareVisualizationSelector().select([])
    except TypeError as exc:
        assert "pandas DataFrame" in str(exc)
    else:
        raise AssertionError("Non-DataFrame input was accepted.")


def test_selector_rejects_invalid_analytics_execution_type() -> None:
    dataframe = pd.DataFrame({"revenue": [100.0]})

    try:
        AnalyticsAwareVisualizationSelector().select(
            dataframe,
            [],
        )
    except TypeError as exc:
        assert "AnalyticsExecutionResult" in str(exc)
    else:
        raise AssertionError("Invalid analytics execution was accepted.")
