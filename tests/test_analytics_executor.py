
from __future__ import annotations

import pandas as pd
import pytest

from datapilot.analytics.contribution import ContributionResult
from datapilot.analytics.executor import AnalyticsExecutor
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.planning import (
    AggregationType,
    AnalysisInputMapping,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlanValidation,
    PlanValidationIssue,
    SkippedAnalysis,
)
from datapilot.analytics.root_cause import RootCauseResult
from datapilot.analytics.segmentation import SegmentationResult
from datapilot.analytics.trend import TrendResult
from datapilot.analytics.variance import VarianceResult


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["June", "June", "July", "July"],
            "revenue": [60.0, 40.0, 50.0, 35.0],
            "region": ["North", "South", "North", "South"],
            "revenue_change": [-5.0, -4.0, -3.0, -3.0],
        }
    )


def request(
    request_id: str,
    analysis_type: AnalysisType,
    inputs: AnalysisInputMapping,
    depends_on: tuple[str, ...] = (),
) -> AnalysisRequest:
    return AnalysisRequest(
        request_id=request_id,
        analysis_type=analysis_type,
        inputs=inputs,
        depends_on=depends_on,
    )


def validation(
    *requests: AnalysisRequest,
    skipped: tuple[SkippedAnalysis, ...] = (),
) -> AnalyticsPlanValidation:
    return AnalyticsPlanValidation(
        accepted_requests=tuple(requests),
        skipped_analyses=skipped,
    )


def test_executes_trend_analysis(dataframe: pd.DataFrame) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "trend",
                AnalysisType.TREND,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="revenue",
                ),
            )
        ),
    )

    result = outcome.get_result("trend")

    assert isinstance(result, TrendResult)
    assert result.first_value == 60.0
    assert result.last_value == 35.0
    assert result.absolute_change == -25.0
    assert len(outcome.executions) == 1
    assert outcome.skipped_analyses == ()


def test_executes_period_comparison_with_sum(
    dataframe: pd.DataFrame,
) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "comparison",
                AnalysisType.PERIOD_COMPARISON,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="revenue",
                    reference_period="June",
                    current_period="July",
                    aggregation=AggregationType.SUM,
                ),
            )
        ),
    )

    result = outcome.get_result("comparison")

    assert isinstance(result, PeriodComparison)
    assert result.value_a == 100.0
    assert result.value_b == 85.0
    assert result.absolute_change == -15.0
    assert result.direction == "decreased"


@pytest.mark.parametrize(
    ("aggregation", "expected_reference", "expected_current"),
    [
        (AggregationType.SUM, 100.0, 85.0),
        (AggregationType.MEAN, 50.0, 42.5),
        (AggregationType.MIN, 40.0, 35.0),
        (AggregationType.MAX, 60.0, 50.0),
        (AggregationType.COUNT, 2.0, 2.0),
    ],
)
def test_period_comparison_supports_all_aggregations(
    dataframe: pd.DataFrame,
    aggregation: AggregationType,
    expected_reference: float,
    expected_current: float,
) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "comparison",
                AnalysisType.PERIOD_COMPARISON,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="revenue",
                    reference_period="June",
                    current_period="July",
                    aggregation=aggregation,
                ),
            )
        ),
    )

    result = outcome.get_result("comparison")

    assert isinstance(result, PeriodComparison)
    assert result.value_a == expected_reference
    assert result.value_b == expected_current


def test_count_aggregation_counts_non_null_values() -> None:
    frame = pd.DataFrame(
        {
            "period": ["June", "June", "June", "July"],
            "description": ["A", None, "C", "D"],
        }
    )

    outcome = AnalyticsExecutor().execute(
        frame,
        validation(
            request(
                "comparison",
                AnalysisType.PERIOD_COMPARISON,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="description",
                    reference_period="June",
                    current_period="July",
                    aggregation=AggregationType.COUNT,
                ),
            )
        ),
    )

    result = outcome.get_result("comparison")

    assert isinstance(result, PeriodComparison)
    assert result.value_a == 2.0
    assert result.value_b == 1.0


def test_executes_variance_analysis(dataframe: pd.DataFrame) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "variance",
                AnalysisType.VARIANCE,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="revenue",
                    reference_period="June",
                    actual_period="July",
                    aggregation=AggregationType.SUM,
                ),
            )
        ),
    )

    result = outcome.get_result("variance")

    assert isinstance(result, VarianceResult)
    assert result.reference_value == 100.0
    assert result.actual_value == 85.0
    assert result.absolute_variance == -15.0


def test_executes_contribution_analysis(
    dataframe: pd.DataFrame,
) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "contribution",
                AnalysisType.CONTRIBUTION,
                AnalysisInputMapping(
                    dimension_column="region",
                    value_column="revenue_change",
                ),
            )
        ),
    )

    result = outcome.get_result("contribution")

    assert isinstance(result, ContributionResult)
    assert result.total_change == -15.0
    assert result.dimension_column == "region"


def test_executes_segmentation_analysis(
    dataframe: pd.DataFrame,
) -> None:
    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            request(
                "segmentation",
                AnalysisType.SEGMENTATION,
                AnalysisInputMapping(
                    dimension_column="region",
                    value_column="revenue",
                ),
            )
        ),
    )

    result = outcome.get_result("segmentation")

    assert isinstance(result, SegmentationResult)
    assert result.total_value == 185.0
    assert len(result.segments) == 2


def test_executes_root_cause_after_dependencies(
    dataframe: pd.DataFrame,
) -> None:
    variance_request = request(
        "variance",
        AnalysisType.VARIANCE,
        AnalysisInputMapping(
            period_column="period",
            value_column="revenue",
            reference_period="June",
            actual_period="July",
            aggregation=AggregationType.SUM,
        ),
    )
    contribution_request = request(
        "contribution",
        AnalysisType.CONTRIBUTION,
        AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue_change",
        ),
    )
    root_cause_request = request(
        "root",
        AnalysisType.ROOT_CAUSE,
        AnalysisInputMapping(),
        depends_on=("variance", "contribution"),
    )

    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            variance_request,
            contribution_request,
            root_cause_request,
        ),
    )

    result = outcome.get_result("root")

    assert isinstance(result, RootCauseResult)
    assert result.overall_change == -15.0
    assert result.dimension_column == "region"
    assert len(outcome.executions) == 3


def test_preserves_validation_skips(
    dataframe: pd.DataFrame,
) -> None:
    skipped = SkippedAnalysis(
        request_id="unsupported",
        analysis_type=AnalysisType.CONTRIBUTION,
        reason="Untrusted change column.",
    )

    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(skipped=(skipped,)),
    )

    assert outcome.executions == ()
    assert outcome.skipped_analyses == (skipped,)


def test_fatal_validation_prevents_execution(
    dataframe: pd.DataFrame,
) -> None:
    fatal = PlanValidationIssue(
        code="DEPENDENCY_CYCLE",
        message="Dependency cycle detected.",
    )

    outcome = AnalyticsExecutor().execute(
        dataframe,
        AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=(),
            fatal_errors=(fatal,),
        ),
    )

    assert outcome.executions == ()
    assert outcome.fatal_errors == (fatal,)


def test_failed_request_is_skipped_without_stopping_independent_work(
    dataframe: pd.DataFrame,
) -> None:
    failed_request = request(
        "bad-trend",
        AnalysisType.TREND,
        AnalysisInputMapping(
            period_column="period",
            value_column="missing_measure",
        ),
    )
    valid_request = request(
        "valid-segmentation",
        AnalysisType.SEGMENTATION,
        AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue",
        ),
    )

    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            failed_request,
            valid_request,
        ),
    )

    assert outcome.get_result("bad-trend") is None
    assert isinstance(
        outcome.get_result("valid-segmentation"),
        SegmentationResult,
    )
    assert len(outcome.skipped_analyses) == 1
    assert outcome.skipped_analyses[0].request_id == "bad-trend"


def test_failed_dependency_skips_dependent_root_cause(
    dataframe: pd.DataFrame,
) -> None:
    variance_request = request(
        "variance",
        AnalysisType.VARIANCE,
        AnalysisInputMapping(
            period_column="period",
            value_column="revenue",
            reference_period="June",
            actual_period="July",
            aggregation=AggregationType.SUM,
        ),
    )
    contribution_request = request(
        "contribution",
        AnalysisType.CONTRIBUTION,
        AnalysisInputMapping(
            dimension_column="region",
            value_column="missing_change",
        ),
    )
    root_request = request(
        "root",
        AnalysisType.ROOT_CAUSE,
        AnalysisInputMapping(),
        depends_on=("variance", "contribution"),
    )

    outcome = AnalyticsExecutor().execute(
        dataframe,
        validation(
            variance_request,
            contribution_request,
            root_request,
        ),
    )

    assert isinstance(
        outcome.get_result("variance"),
        VarianceResult,
    )
    assert outcome.get_result("contribution") is None
    assert outcome.get_result("root") is None

    skipped_ids = {
        item.request_id
        for item in outcome.skipped_analyses
    }

    assert skipped_ids == {"contribution", "root"}


def test_empty_numeric_period_is_skipped() -> None:
    frame = pd.DataFrame(
        {
            "period": ["June", "July"],
            "revenue": [None, 100.0],
        }
    )

    outcome = AnalyticsExecutor().execute(
        frame,
        validation(
            request(
                "variance",
                AnalysisType.VARIANCE,
                AnalysisInputMapping(
                    period_column="period",
                    value_column="revenue",
                    reference_period="June",
                    actual_period="July",
                    aggregation=AggregationType.SUM,
                ),
            )
        ),
    )

    assert outcome.get_result("variance") is None
    assert len(outcome.skipped_analyses) == 1
    assert "no valid numeric observations" in (
        outcome.skipped_analyses[0].reason
    )


def test_rejects_invalid_dataframe_type() -> None:
    with pytest.raises(TypeError, match="Dataframe"):
        AnalyticsExecutor().execute(
            [],
            validation(),
        )


def test_rejects_invalid_validation_type(
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="AnalyticsPlanValidation"):
        AnalyticsExecutor().execute(
            dataframe,
            None,
        )
