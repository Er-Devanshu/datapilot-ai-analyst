import pandas as pd
import pytest

from datapilot.analytics.answer_validation import AnswerValidator
from datapilot.analytics.contribution import (
    Contribution,
    ContributionResult,
)
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.root_cause import RootCauseAnalyzer
from datapilot.analytics.trend import TrendPoint, TrendResult
from datapilot.analytics.variance import VarianceResult


def _directional_evidence(direction: str):
    dataframe = pd.DataFrame({"revenue": [100.0, 80.0]})

    return EvidenceBuilder().build(
        dataframe=dataframe,
        trend=TrendResult(
            period_column="month",
            value_column="revenue",
            points=(
                TrendPoint("Jan", 100.0),
                TrendPoint("Feb", 80.0),
            ),
            first_value=100.0,
            last_value=80.0,
            absolute_change=-20.0,
            percentage_change=-20.0,
            direction=direction,
        ),
    )


def _root_cause_evidence():
    dataframe = pd.DataFrame(
        {
            "region": ["North", "South"],
            "change": [20.0, -5.0],
        }
    )

    contribution = ContributionResult(
        dimension_column="region",
        value_column="change",
        total_change=15.0,
        contributions=(
            Contribution("North", 20.0, 133.333333),
            Contribution("South", -5.0, -33.333333),
        ),
    )

    variance = VarianceResult(
        actual_value=115.0,
        reference_value=100.0,
        absolute_variance=15.0,
        percentage_variance=15.0,
        direction="above_reference",
    )

    root_cause = RootCauseAnalyzer().analyze(
        variance=variance,
        contribution=contribution,
    )

    return EvidenceBuilder().build(
        dataframe=dataframe,
        root_cause=root_cause,
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue did not increase.",
        "Revenue never increased.",
        "Revenue did not rise.",
        "Revenue was not higher.",
    ],
)
def test_negated_positive_direction_does_not_trigger_warning(
    answer: str,
) -> None:
    result = AnswerValidator().validate(
        answer,
        _directional_evidence("decreasing"),
    )

    assert not any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue did not decrease.",
        "Revenue never declined.",
        "Revenue was not lower.",
    ],
)
def test_negated_negative_direction_does_not_trigger_warning(
    answer: str,
) -> None:
    result = AnswerValidator().validate(
        answer,
        _directional_evidence("increasing"),
    )

    assert not any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


def test_unnegated_directional_contradiction_still_triggers_warning():
    result = AnswerValidator().validate(
        "Revenue increased.",
        _directional_evidence("decreasing"),
    )

    assert any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The North region was not responsible for the increase.",
        "The increase was not caused by the North region.",
        "The North region never caused the increase.",
    ],
)
def test_negated_causal_claim_does_not_trigger_warning(
    answer: str,
) -> None:
    result = AnswerValidator().validate(
        answer,
        _root_cause_evidence(),
    )

    assert not any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "answer",
    [
        'The report states, "North caused the increase."',
        'The phrase "resulted in" appears in the source text.',
        "The report says “the root cause was North.”",
    ],
)
def test_quoted_causal_language_does_not_trigger_warning(
    answer: str,
) -> None:
    result = AnswerValidator().validate(
        answer,
        _root_cause_evidence(),
    )

    assert not any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )


def test_unquoted_causal_claim_still_triggers_warning():
    result = AnswerValidator().validate(
        "The North region caused the increase.",
        _root_cause_evidence(),
    )

    assert any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )


def test_quoted_directional_contradiction_does_not_trigger_warning():
    result = AnswerValidator().validate(
        'The report says, "Revenue increased."',
        _directional_evidence("decreasing"),
    )

    assert not any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )