
from __future__ import annotations

import pandas as pd
import pytest

from datapilot.analytics.contribution import ContributionAnalyzer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.executor import (
    AnalysisExecution,
    AnalyticsExecutionResult,
)
from datapilot.analytics.period import PeriodComparator
from datapilot.analytics.planning import (
    AnalysisType,
    PlanValidationIssue,
    SkippedAnalysis,
)
from datapilot.analytics.root_cause import RootCauseAnalyzer
from datapilot.analytics.segmentation import SegmentationAnalyzer
from datapilot.analytics.trend import TrendAnalyzer
from datapilot.analytics.variance import VarianceAnalyzer


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar"],
            "region": ["North", "West", "East"],
            "revenue": [100.0, 120.0, 150.0],
            "revenue_change": [-7.0, -5.0, -3.0],
        }
    )


def _build_all_executions() -> tuple[AnalysisExecution, ...]:
    dataframe = _build_dataframe()

    trend = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    period_comparison = PeriodComparator().compare(
        value_a=100,
        value_b=125,
    )

    variance = VarianceAnalyzer().analyze(
        actual_value=85,
        reference_value=100,
    )

    contribution = ContributionAnalyzer().analyze_change(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    segmentation = SegmentationAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue",
    )

    root_cause = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    return (
        AnalysisExecution("trend_1", AnalysisType.TREND, trend),
        AnalysisExecution(
            "comparison_1",
            AnalysisType.PERIOD_COMPARISON,
            period_comparison,
        ),
        AnalysisExecution("variance_1", AnalysisType.VARIANCE, variance),
        AnalysisExecution(
            "contribution_1",
            AnalysisType.CONTRIBUTION,
            contribution,
        ),
        AnalysisExecution(
            "segmentation_1",
            AnalysisType.SEGMENTATION,
            segmentation,
        ),
        AnalysisExecution(
            "root_cause_1",
            AnalysisType.ROOT_CAUSE,
            root_cause,
        ),
    )


def test_all_six_analytics_result_types_are_structured() -> None:
    dataframe = _build_dataframe()
    executions = _build_all_executions()

    execution_result = AnalyticsExecutionResult(
        executions=executions,
        skipped_analyses=(),
        fatal_errors=(),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    prompt_text = evidence.to_prompt_text()

    expected_types = (
        "TrendResult",
        "PeriodComparison",
        "VarianceResult",
        "ContributionResult",
        "SegmentationResult",
        "RootCauseResult",
    )

    for result_type in expected_types:
        assert f"Result type: {result_type}" in prompt_text

    assert "Deterministic result:" not in prompt_text
    assert "revenue_change" in prompt_text
    assert "Drivers (contribution decomposition, not proven causes):" in (
        prompt_text
    )


def test_multiple_same_type_executions_preserve_order_and_request_ids() -> None:
    dataframe = _build_dataframe()

    first = VarianceAnalyzer().analyze(
        actual_value=90,
        reference_value=100,
    )
    second = VarianceAnalyzer().analyze(
        actual_value=80,
        reference_value=100,
    )

    executions = (
        AnalysisExecution("variance_first", AnalysisType.VARIANCE, first),
        AnalysisExecution("variance_second", AnalysisType.VARIANCE, second),
    )

    execution_result = AnalyticsExecutionResult(
        executions=executions,
        skipped_analyses=(),
        fatal_errors=(),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    prompt_text = evidence.to_prompt_text()

    first_position = prompt_text.index("variance_first")
    second_position = prompt_text.index("variance_second")

    assert first_position < second_position
    assert prompt_text.count("Result type: VarianceResult") == 2
    assert prompt_text.count("Absolute variance:") == 2


def test_skipped_analyses_preserve_request_and_reason() -> None:
    dataframe = _build_dataframe()

    skipped = SkippedAnalysis(
        request_id="unsupported_trend",
        analysis_type=AnalysisType.TREND,
        reason="Required period column is unavailable.",
    )

    execution_result = AnalyticsExecutionResult(
        executions=(),
        skipped_analyses=(skipped,),
        fatal_errors=(),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    prompt_text = evidence.to_prompt_text()

    assert "SKIPPED ANALYSES:" in prompt_text
    assert "unsupported_trend" in prompt_text
    assert "Required period column is unavailable." in prompt_text


def test_fatal_errors_preserve_structured_validation_details() -> None:
    dataframe = _build_dataframe()

    issue = PlanValidationIssue(
        code="invalid_plan",
        message="The analytics plan contains an invalid dependency.",
        request_id="request_123",
    )

    execution_result = AnalyticsExecutionResult(
        executions=(),
        skipped_analyses=(),
        fatal_errors=(issue,),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    prompt_text = evidence.to_prompt_text()

    assert "FATAL ANALYTICS ERRORS:" in prompt_text
    assert "Code: invalid_plan" in prompt_text
    assert "The analytics plan contains an invalid dependency." in prompt_text
    assert "Request ID: request_123" in prompt_text


def test_none_percentage_is_preserved_as_unavailable() -> None:
    dataframe = _build_dataframe()

    variance = VarianceAnalyzer().analyze(
        actual_value=0,
        reference_value=0,
    )

    execution_result = AnalyticsExecutionResult(
        executions=(
            AnalysisExecution(
                "zero_baseline_variance",
                AnalysisType.VARIANCE,
                variance,
            ),
        ),
        skipped_analyses=(),
        fatal_errors=(),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    prompt_text = evidence.to_prompt_text()

    assert variance.percentage_variance is None
    assert "Percentage variance: None" in prompt_text
    assert "Percentage variance: 0.0" not in prompt_text


def test_sql_dataframe_summaries_remain_unchanged() -> None:
    dataframe = _build_dataframe()

    execution_result = AnalyticsExecutionResult(
        executions=_build_all_executions(),
        skipped_analyses=(),
        fatal_errors=(),
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analytics_execution_result=execution_result,
    )

    assert evidence.row_count == 3
    assert evidence.result_columns == (
        "month",
        "region",
        "revenue",
        "revenue_change",
    )

    revenue_summary = next(
        item
        for item in evidence.numeric_summaries
        if item.column == "revenue"
    )

    assert revenue_summary.total == pytest.approx(370.0)
    assert revenue_summary.average == pytest.approx(370.0 / 3.0)


def test_evidence_builder_rejects_both_execution_input_forms() -> None:
    dataframe = _build_dataframe()
    execution = _build_all_executions()[0]

    execution_result = AnalyticsExecutionResult(
        executions=(execution,),
        skipped_analyses=(),
        fatal_errors=(),
    )

    with pytest.raises(ValueError, match="not both"):
        EvidenceBuilder().build(
            dataframe=dataframe,
            analytics_executions=(execution,),
            analytics_execution_result=execution_result,
        )
