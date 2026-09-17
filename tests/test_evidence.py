import pandas as pd
import pytest

from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.evidence import EvidenceBuilder


def test_evidence_builds_from_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "West"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    analysis = ResultAnalyzer().analyze(dataframe)

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        analysis=analysis,
    )

    assert evidence.row_count == 3
    assert evidence.result_columns == (
        "region",
        "revenue",
    )

    assert len(evidence.numeric_summaries) == 1
    assert evidence.numeric_summaries[0].column == "revenue"


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
    assert len(evidence.numeric_summaries) == 1


def test_evidence_prompt_text_contains_grounded_facts() -> None:
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
    )

    prompt_text = evidence.to_prompt_text()

    assert "Result rows: 3" in prompt_text
    assert "Column: revenue" in prompt_text
    assert "Total: 600.0" in prompt_text
    assert "Average: 200.0" in prompt_text
    assert "Minimum: 100.0" in prompt_text
    assert "Maximum: 300.0" in prompt_text
    assert "revenue" in prompt_text