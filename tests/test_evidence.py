import pandas as pd
import pytest

from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.period import PeriodComparator
from datapilot.analytics.trend import TrendAnalyzer


def test_evidence_builds_from_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "West"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    analysis = ResultAnalyzer().analyze(
        dataframe
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analysis=analysis,
    )

    assert evidence.row_count == 3
    assert evidence.result_columns == (
        "region",
        "revenue",
    )
    assert evidence.currency == "INR"

    assert len(
        evidence.numeric_summaries
    ) == 1

    assert (
        evidence.numeric_summaries[0].column
        == "revenue"
    )

    assert len(
        evidence.category_summaries
    ) == 1

    category_summary = (
        evidence.category_summaries[0]
    )

    assert category_summary.dimension == "region"
    assert category_summary.measure == "revenue"
    assert category_summary.top_category == "West"
    assert category_summary.top_value == 400.0
    assert category_summary.bottom_category == "East"
    assert category_summary.bottom_value == 200.0


def test_evidence_rejects_mismatched_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    analysis = ResultAnalyzer().analyze(
        pd.DataFrame(
            {
                "revenue": [100.0],
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="row count does not match",
    ):
        EvidenceBuilder().build(
            dataframe=dataframe,
            analysis=analysis,
        )


def test_evidence_can_analyze_without_precomputed_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    assert evidence.row_count == 3
    assert len(
        evidence.numeric_summaries
    ) == 1

    summary = (
        evidence.numeric_summaries[0]
    )

    assert summary.column == "revenue"
    assert summary.total == 600.0
    assert summary.average == 200.0
    assert summary.minimum == 100.0
    assert summary.maximum == 300.0


def test_evidence_prompt_text_contains_grounded_facts() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "West"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    prompt_text = evidence.to_prompt_text()

    assert "EVIDENCE:" in prompt_text
    assert "Result rows: 3" in prompt_text
    assert "Currency: INR" in prompt_text

    assert "Column: revenue" in prompt_text
    assert "Total: 600.0" in prompt_text
    assert "Average: 200.0" in prompt_text
    assert "Minimum: 100.0" in prompt_text
    assert "Maximum: 300.0" in prompt_text

    assert "Dimension: region" in prompt_text
    assert "Measure: revenue" in prompt_text
    assert "Top category: West" in prompt_text
    assert "Top value: 400.0" in prompt_text
    assert "Bottom category: East" in prompt_text
    assert "Bottom value: 200.0" in prompt_text

    assert "TREND ANALYSIS:" in prompt_text
    assert "PERIOD COMPARISON:" in prompt_text
    assert "RESULT COLUMNS:" in prompt_text
    assert "region, revenue" in prompt_text


def test_evidence_defaults_to_inr_currency() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    assert evidence.currency == "INR"

    assert (
        "Currency: INR"
        in evidence.to_prompt_text()
    )


def test_evidence_supports_explicit_currency() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0],
        }
    )

    evidence = EvidenceBuilder(
        currency="USD"
    ).build(
        dataframe=dataframe,
    )

    assert evidence.currency == "USD"

    assert (
        "Currency: USD"
        in evidence.to_prompt_text()
    )


def test_evidence_rejects_empty_currency() -> None:
    with pytest.raises(
        ValueError,
        match="Currency cannot be empty",
    ):
        EvidenceBuilder(
            currency=" "
        )


def test_evidence_includes_trend_analysis() -> None:
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

    prompt_text = evidence.to_prompt_text()

    assert "Direction: increasing" in prompt_text
    assert "First value: 100.0" in prompt_text
    assert "Last value: 150.0" in prompt_text
    assert "Absolute change: 50.0" in prompt_text
    assert "Percentage change: 50.0" in prompt_text


def test_evidence_includes_period_comparison() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [
                100.0,
                125.0,
            ],
        }
    )

    comparison = PeriodComparator().compare(
        value_a=100.0,
        value_b=125.0,
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        period_comparison=comparison,
    )

    assert (
        evidence.period_comparison
        is comparison
    )

    prompt_text = evidence.to_prompt_text()

    assert "Reference value: 100.0" in prompt_text
    assert "Current value: 125.0" in prompt_text
    assert "Absolute change: 25.0" in prompt_text
    assert "Percentage change: 25.0" in prompt_text
    assert "Direction: increased" in prompt_text