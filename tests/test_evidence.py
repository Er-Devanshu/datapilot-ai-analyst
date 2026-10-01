from __future__ import annotations

import pandas as pd
import pytest

from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.contribution import ContributionAnalyzer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.period import PeriodComparator
from datapilot.analytics.root_cause import RootCauseAnalyzer
from datapilot.analytics.segmentation import SegmentationAnalyzer
from datapilot.analytics.trend import TrendAnalyzer
from datapilot.analytics.variance import VarianceAnalyzer


def test_evidence_builds_from_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "West"],
            "revenue": [400, 200, 300],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    assert evidence.row_count == 3
    assert evidence.result_columns == (
        "region",
        "revenue",
    )
    assert evidence.currency == "INR"

    assert len(evidence.numeric_summaries) == 1
    assert evidence.numeric_summaries[0].column == "revenue"
    assert evidence.numeric_summaries[0].total == 900

    assert len(evidence.category_summaries) == 1
    assert evidence.category_summaries[0].dimension == "region"
    assert evidence.category_summaries[0].measure == "revenue"
    assert evidence.category_summaries[0].top_category == "West"
    assert evidence.category_summaries[0].top_value == 700
    assert evidence.category_summaries[0].bottom_category == "East"
    assert evidence.category_summaries[0].bottom_value == 200


def test_evidence_rejects_mismatched_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    analysis_dataframe = pd.DataFrame(
        {
            "region": ["West"],
            "revenue": [400],
        }
    )

    analysis = ResultAnalyzer().analyze(
        analysis_dataframe
    )

    with pytest.raises(ValueError):
        EvidenceBuilder().build(
            dataframe=dataframe,
            analysis=analysis,
        )


def test_evidence_can_analyze_without_precomputed_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    assert evidence.row_count == 2
    assert evidence.numeric_summaries[0].total == 600


def test_evidence_prompt_contains_grounded_facts() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    prompt_text = evidence.to_prompt_text()

    assert "EVIDENCE:" in prompt_text
    assert "Result rows: 2" in prompt_text
    assert "Currency: INR" in prompt_text
    assert "Column: revenue" in prompt_text
    assert "Total: 600.0" in prompt_text
    assert "Top category: West" in prompt_text
    assert "Top value: 400.0" in prompt_text
    assert "Bottom category: East" in prompt_text
    assert "Bottom value: 200.0" in prompt_text
    assert "RESULT COLUMNS:" in prompt_text
    assert "region, revenue" in prompt_text


def test_evidence_defaults_to_inr() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    assert evidence.currency == "INR"


def test_evidence_supports_explicit_usd() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100],
        }
    )

    evidence = EvidenceBuilder(
        currency="USD"
    ).build(
        dataframe=dataframe,
    )

    assert evidence.currency == "USD"
    assert "Currency: USD" in evidence.to_prompt_text()


def test_evidence_rejects_empty_currency() -> None:
    with pytest.raises(ValueError):
        EvidenceBuilder(currency="")


def test_evidence_includes_trend() -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar"],
            "revenue": [100, 120, 150],
        }
    )

    trend = TrendAnalyzer().analyze(
        dataframe=dataframe,
        period_column="month",
        value_column="revenue",
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        trend=trend,
    )

    assert evidence.trend is trend
    assert evidence.trend.direction == "increasing"

    prompt_text = evidence.to_prompt_text()

    assert "TREND ANALYSIS:" in prompt_text
    assert "Period column: month" in prompt_text
    assert "Value column: revenue" in prompt_text
    assert "First value: 100.0" in prompt_text
    assert "Last value: 150.0" in prompt_text
    assert "Absolute change: 50.0" in prompt_text
    assert "Percentage change: 50.0" in prompt_text
    assert "Direction: increasing" in prompt_text


def test_evidence_includes_period_comparison() -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["June", "July"],
            "revenue": [100, 125],
        }
    )

    comparison = PeriodComparator().compare(
        value_a=100,
        value_b=125,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        period_comparison=comparison,
    )

    assert evidence.period_comparison is comparison

    prompt_text = evidence.to_prompt_text()

    assert "PERIOD COMPARISON:" in prompt_text
    assert "Reference value: 100.0" in prompt_text
    assert "Current value: 125.0" in prompt_text
    assert "Absolute change: 25.0" in prompt_text
    assert "Percentage change: 25.0" in prompt_text
    assert "Direction: increased" in prompt_text


def test_evidence_includes_variance() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    variance = VarianceAnalyzer().analyze(
        actual_value=1200,
        reference_value=1000,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        variance=variance,
    )

    assert evidence.variance is variance
    assert evidence.variance.absolute_variance == 200
    assert evidence.variance.percentage_variance == 20
    assert evidence.variance.direction == "above_reference"


def test_evidence_prompt_contains_variance() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East"],
            "revenue": [400, 200],
        }
    )

    variance = VarianceAnalyzer().analyze(
        actual_value=1200,
        reference_value=1000,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        variance=variance,
    )

    prompt_text = evidence.to_prompt_text()

    assert "VARIANCE ANALYSIS:" in prompt_text
    assert "Actual value: 1200.0" in prompt_text
    assert "Reference value: 1000.0" in prompt_text
    assert "Absolute variance: 200.0" in prompt_text
    assert "Percentage variance: 20.0" in prompt_text
    assert "Direction: above_reference" in prompt_text


def test_evidence_includes_contribution() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
                "South",
                "East",
            ],
            "revenue_change": [
                -80,
                -50,
                -30,
                10,
            ],
        }
    )

    contribution = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        contribution=contribution,
    )

    assert evidence.contribution is contribution
    assert evidence.contribution.total_change == -150
    assert len(
        evidence.contribution.contributions
    ) == 4


def test_evidence_prompt_contains_contribution() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
                "South",
                "East",
            ],
            "revenue_change": [
                -80,
                -50,
                -30,
                10,
            ],
        }
    )

    contribution = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        contribution=contribution,
    )

    prompt_text = evidence.to_prompt_text()

    assert "CONTRIBUTION ANALYSIS:" in prompt_text
    assert "Dimension column: region" in prompt_text
    assert "Value column: revenue_change" in prompt_text
    assert "Total change: -150.0" in prompt_text
    assert "Contributions:" in prompt_text
    assert "Category: East" in prompt_text
    assert "Category: North" in prompt_text


def test_evidence_includes_segmentation() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_segment": [
                "Enterprise",
                "Enterprise",
                "SMB",
                "Consumer",
            ],
            "revenue": [
                500,
                300,
                200,
                100,
            ],
        }
    )

    segmentation = SegmentationAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="customer_segment",
        value_column="revenue",
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        segmentation=segmentation,
    )

    assert evidence.segmentation is segmentation
    assert evidence.segmentation.total_value == 1100
    assert len(
        evidence.segmentation.segments
    ) == 3
    assert (
        evidence.segmentation.segments[0].segment
        == "Enterprise"
    )


def test_evidence_prompt_contains_segmentation() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_segment": [
                "Enterprise",
                "Enterprise",
                "SMB",
                "Consumer",
            ],
            "revenue": [
                500,
                300,
                200,
                100,
            ],
        }
    )

    segmentation = SegmentationAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="customer_segment",
        value_column="revenue",
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        segmentation=segmentation,
    )

    prompt_text = evidence.to_prompt_text()

    assert "SEGMENTATION ANALYSIS:" in prompt_text
    assert "Dimension column: customer_segment" in prompt_text
    assert "Value column: revenue" in prompt_text
    assert "Total value: 1100.0" in prompt_text
    assert "Segments:" in prompt_text
    assert "Segment: Enterprise" in prompt_text
    assert "Total value: 800.0" in prompt_text
    assert "Average value: 400.0" in prompt_text


def test_evidence_includes_root_cause() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
                "South",
                "East",
            ],
            "revenue_change": [
                -8,
                -5,
                -3,
                1,
            ],
        }
    )

    variance = VarianceAnalyzer().analyze(
        actual_value=85,
        reference_value=100,
    )

    contribution = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    root_cause = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        variance=variance,
        contribution=contribution,
        root_cause=root_cause,
    )

    assert evidence.root_cause is root_cause
    assert evidence.root_cause.overall_change == -15
    assert (
        evidence.root_cause.overall_direction
        == "below_reference"
    )
    assert len(
        evidence.root_cause.drivers
    ) == 4


def test_evidence_prompt_contains_root_cause() -> None:
    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "West",
                "South",
                "East",
            ],
            "revenue_change": [
                -8,
                -5,
                -3,
                1,
            ],
        }
    )

    variance = VarianceAnalyzer().analyze(
        actual_value=85,
        reference_value=100,
    )

    contribution = ContributionAnalyzer().analyze(
        dataframe=dataframe,
        dimension_column="region",
        value_column="revenue_change",
    )

    root_cause = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        variance=variance,
        contribution=contribution,
        root_cause=root_cause,
    )

    prompt_text = evidence.to_prompt_text()

    assert "ROOT-CAUSE ANALYSIS:" in prompt_text
    assert "Dimension column: region" in prompt_text
    assert "Value column: revenue_change" in prompt_text
    assert "Overall change: -15.0" in prompt_text
    assert "Overall direction: below_reference" in prompt_text
    assert "Drivers:" in prompt_text
    assert "Category: North" in prompt_text
    assert "Direction: negative" in prompt_text
    assert "Category: East" in prompt_text
    assert "Direction: positive" in prompt_text


def test_evidence_includes_visualization() -> None:
    from datapilot.visualization.selector import VisualizationSelector

    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "West",
            ],
            "revenue": [
                100,
                200,
                150,
            ],
        }
    )

    visualization = VisualizationSelector().select(
        dataframe=dataframe,
        title="Revenue by Region",
    ).specification

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        visualization=visualization,
    )

    assert evidence.visualization is visualization
    assert evidence.visualization.chart_type == "bar"
    assert evidence.visualization.x_column == "region"
    assert evidence.visualization.y_column == "revenue"


def test_evidence_prompt_contains_visualization() -> None:
    from datapilot.visualization.selector import VisualizationSelector

    dataframe = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
                "West",
            ],
            "revenue": [
                100,
                200,
                150,
            ],
        }
    )

    visualization = VisualizationSelector().select(
        dataframe=dataframe,
        title="Revenue by Region",
    ).specification

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        visualization=visualization,
    )

    prompt_text = evidence.to_prompt_text()

    assert "VISUALIZATION:" in prompt_text
    assert "Chart type: bar" in prompt_text
    assert "X column: region" in prompt_text
    assert "Y column: revenue" in prompt_text
    assert "Title: Revenue by Region" in prompt_text