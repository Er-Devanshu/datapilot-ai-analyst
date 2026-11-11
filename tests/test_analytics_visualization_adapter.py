
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from datapilot.analytics.contribution import (
    Contribution,
    ContributionResult,
)
from datapilot.analytics.executor import AnalysisExecution
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.planning import AnalysisType
from datapilot.analytics.root_cause import (
    RootCauseDriver,
    RootCauseResult,
)
from datapilot.analytics.segmentation import (
    SegmentationResult,
    SegmentSummary,
)
from datapilot.analytics.trend import TrendPoint, TrendResult
from datapilot.analytics.variance import VarianceResult
from datapilot.visualization.analytics_adapter import (
    AnalyticsVisualizationAdapter,
)


@pytest.fixture
def adapter() -> AnalyticsVisualizationAdapter:
    return AnalyticsVisualizationAdapter()


def make_execution(
    analysis_type: AnalysisType,
    result: object,
    request_id: str = "analysis_1",
) -> AnalysisExecution:
    return AnalysisExecution(
        request_id=request_id,
        analysis_type=analysis_type,
        result=result,
    )


def test_adapts_trend_result(adapter):
    result = TrendResult(
        period_column="month",
        value_column="revenue",
        points=(
            TrendPoint(period="2026-01", value=100.0),
            TrendPoint(period="2026-02", value=150.0),
            TrendPoint(period="2026-03", value=175.0),
        ),
        first_value=100.0,
        last_value=175.0,
        absolute_change=75.0,
        percentage_change=75.0,
        direction="increasing",
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.TREND, result)
    )

    assert adapted is not None
    assert adapted.request_id == "analysis_1"
    assert adapted.analysis_type == "trend"
    assert adapted.chart_type == "line"
    assert adapted.x_column == "period"
    assert adapted.y_column == "value"
    assert adapted.title == "revenue Trend"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "period": ["2026-01", "2026-02", "2026-03"],
                "value": [100.0, 150.0, 175.0],
            }
        ),
    )


def test_adapts_segmentation_result(adapter):
    result = SegmentationResult(
        dimension_column="department",
        value_column="revenue",
        total_value=300.0,
        segments=(
            SegmentSummary(
                segment="Sales",
                row_count=3,
                total_value=200.0,
                average_value=200.0 / 3,
                minimum_value=50.0,
                maximum_value=100.0,
            ),
            SegmentSummary(
                segment="Marketing",
                row_count=2,
                total_value=100.0,
                average_value=50.0,
                minimum_value=40.0,
                maximum_value=60.0,
            ),
        ),
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.SEGMENTATION, result)
    )

    assert adapted is not None
    assert adapted.chart_type == "bar"
    assert adapted.x_column == "segment"
    assert adapted.y_column == "total_value"
    assert adapted.title == "revenue by department"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "segment": ["Sales", "Marketing"],
                "total_value": [200.0, 100.0],
            }
        ),
    )


def test_adapts_contribution_result(adapter):
    result = ContributionResult(
        dimension_column="region",
        value_column="revenue_change",
        total_change=50.0,
        contributions=(
            Contribution(
                category="North",
                value=80.0,
                contribution_percentage=160.0,
            ),
            Contribution(
                category="South",
                value=-30.0,
                contribution_percentage=-60.0,
            ),
        ),
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.CONTRIBUTION, result)
    )

    assert adapted is not None
    assert adapted.chart_type == "bar"
    assert adapted.x_column == "category"
    assert adapted.y_column == "value"
    assert adapted.title == "Contribution to revenue_change Change"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "category": ["North", "South"],
                "value": [80.0, -30.0],
            }
        ),
    )


def test_adapts_root_cause_result(adapter):
    result = RootCauseResult(
        dimension_column="region",
        value_column="revenue",
        overall_change=50.0,
        overall_direction="increased",
        drivers=(
            RootCauseDriver(
                category="North",
                value=80.0,
                contribution_percentage=160.0,
                direction="positive",
            ),
            RootCauseDriver(
                category="South",
                value=-30.0,
                contribution_percentage=-60.0,
                direction="negative",
            ),
        ),
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.ROOT_CAUSE, result)
    )

    assert adapted is not None
    assert adapted.chart_type == "bar"
    assert adapted.x_column == "category"
    assert adapted.y_column == "value"
    assert adapted.color_column == "direction"
    assert adapted.title == "Drivers of revenue Change"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "category": ["North", "South"],
                "value": [80.0, -30.0],
                "direction": ["positive", "negative"],
            }
        ),
    )


def test_adapts_period_comparison_result(adapter):
    result = PeriodComparison(
        value_a=100.0,
        value_b=125.0,
        absolute_change=25.0,
        percentage_change=25.0,
        direction="increased",
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.PERIOD_COMPARISON, result)
    )

    assert adapted is not None
    assert adapted.chart_type == "table"
    assert adapted.x_column is None
    assert adapted.y_column is None
    assert adapted.title == "Period Comparison"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "metric": [
                    "Reference value",
                    "Current value",
                    "Absolute change",
                    "Percentage change",
                ],
                "value": [100.0, 125.0, 25.0, 25.0],
            }
        ),
    )


def test_adapts_variance_result(adapter):
    result = VarianceResult(
        actual_value=125.0,
        reference_value=100.0,
        absolute_variance=25.0,
        percentage_variance=25.0,
        direction="above_reference",
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.VARIANCE, result)
    )

    assert adapted is not None
    assert adapted.chart_type == "table"
    assert adapted.x_column is None
    assert adapted.y_column is None
    assert adapted.title == "Variance Analysis"

    assert_frame_equal(
        adapted.dataframe,
        pd.DataFrame(
            {
                "metric": [
                    "Actual value",
                    "Reference value",
                    "Absolute variance",
                    "Percentage variance",
                ],
                "value": [125.0, 100.0, 25.0, 25.0],
            }
        ),
    )


def test_preserves_none_percentage_values(adapter):
    result = PeriodComparison(
        value_a=0.0,
        value_b=100.0,
        absolute_change=100.0,
        percentage_change=None,
        direction="increased",
    )

    adapted = adapter.adapt(
        make_execution(AnalysisType.PERIOD_COMPARISON, result)
    )

    assert adapted is not None
    assert pd.isna(adapted.dataframe.loc[3, "value"])


def test_preserves_request_id(adapter):
    result = TrendResult(
        period_column="month",
        value_column="revenue",
        points=(TrendPoint(period="2026-01", value=100.0),),
        first_value=100.0,
        last_value=100.0,
        absolute_change=0.0,
        percentage_change=0.0,
        direction="stable",
    )

    adapted = adapter.adapt(
        make_execution(
            AnalysisType.TREND,
            result,
            request_id="monthly_revenue_trend",
        )
    )

    assert adapted is not None
    assert adapted.request_id == "monthly_revenue_trend"


def test_rejects_invalid_execution_type(adapter):
    with pytest.raises(
        TypeError,
        match="Execution must be an AnalysisExecution",
    ):
        adapter.adapt(object())


def test_returns_none_for_unsupported_result(adapter):
    execution = make_execution(
        AnalysisType.TREND,
        object(),
    )

    assert adapter.adapt(execution) is None
