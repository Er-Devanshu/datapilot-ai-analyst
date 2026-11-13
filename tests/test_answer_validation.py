
from decimal import Decimal

import pandas as pd
import pytest

from datapilot.analytics.answer_validation import (
    AnswerValidationIssue,
    AnswerValidationResult,
    AnswerValidator,
    extract_answer_numbers,
    extract_evidence_numeric_values,
)
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.trend import TrendPoint, TrendResult
from datapilot.analytics.variance import VarianceResult


def _evidence(currency="INR"):
    dataframe = pd.DataFrame(
        {
            "revenue": [100.0, 120.0],
            "region": ["North", "South"],
        }
    )

    return EvidenceBuilder(currency=currency).build(
        dataframe=dataframe,
        trend=TrendResult(
            period_column="month",
            value_column="revenue",
            points=(
                TrendPoint("Jan", 100.0),
                TrendPoint("Feb", 120.0),
            ),
            first_value=100.0,
            last_value=120.0,
            absolute_change=20.0,
            percentage_change=20.0,
            direction="increasing",
        ),
        period_comparison=PeriodComparison(
            value_a=100.0,
            value_b=120.0,
            absolute_change=20.0,
            percentage_change=20.0,
            direction="increased",
        ),
        variance=VarianceResult(
            actual_value=85.0,
            reference_value=100.0,
            absolute_variance=-15.0,
            percentage_variance=-15.0,
            direction="below_reference",
        ),
    )


def test_validation_issue_requires_code_and_message():
    with pytest.raises(ValueError):
        AnswerValidationIssue(code="", message="Invalid value")

    with pytest.raises(ValueError):
        AnswerValidationIssue(code="test", message="")


def test_validation_issue_accepts_warning_and_error():
    warning = AnswerValidationIssue(
        code="unsupported_number",
        message="Number is not supported by evidence.",
    )
    error = AnswerValidationIssue(
        code="invalid_currency",
        message="Currency does not match evidence.",
        severity="error",
    )

    assert warning.severity == "warning"
    assert error.severity == "error"


def test_validation_result_success_is_valid():
    result = AnswerValidationResult.success()

    assert result.valid is True
    assert result.issues == ()


def test_validation_result_with_issues_is_invalid():
    issue = AnswerValidationIssue(
        code="unsupported_number",
        message="Unsupported number.",
    )

    result = AnswerValidationResult.with_issues((issue,))

    assert result.valid is False
    assert result.issues == (issue,)


def test_valid_result_cannot_contain_issues():
    issue = AnswerValidationIssue(
        code="unsupported_number",
        message="Unsupported number.",
    )

    with pytest.raises(ValueError):
        AnswerValidationResult(valid=True, issues=(issue,))


def test_validation_result_requires_tuple():
    with pytest.raises(TypeError):
        AnswerValidationResult(valid=False, issues=[])


def test_extracts_absolute_and_percentage_evidence_separately():
    values = extract_evidence_numeric_values(_evidence())

    assert Decimal("-15.0") in values.absolute_values
    assert Decimal("-15.0") in values.percentage_values
    assert Decimal("20.0") in values.absolute_values
    assert Decimal("20.0") in values.percentage_values


def test_extracts_numbers_with_commas_and_decimals():
    extracted = extract_answer_numbers(
        "Revenue was 1,200.50 and increased by 15.25%."
    )

    assert extracted == (
        ("1,200.50", Decimal("1200.50"), False),
        ("15.25%", Decimal("15.25"), True),
    )


def test_extracts_negative_numbers_and_percentages():
    extracted = extract_answer_numbers(
        "Variance was -15.0 and percentage variance was -15.0%."
    )

    assert extracted == (
        ("-15.0", Decimal("-15.0"), False),
        ("-15.0%", Decimal("-15.0"), True),
    )


def test_ignores_date_components():
    extracted = extract_answer_numbers(
        "The report covers 2026-11-13 and revenue is 600."
    )

    assert extracted == (
        ("600", Decimal("600"), False),
    )


def test_ignores_hyphenated_identifier_suffixes():
    extracted = extract_answer_numbers(
        "Request REQ-123 completed with revenue of 600."
    )

    assert extracted == (
        ("600", Decimal("600"), False),
    )


def test_deduplicates_numeric_evidence_values():
    values = extract_evidence_numeric_values(_evidence())

    assert isinstance(values.absolute_values, frozenset)
    assert isinstance(values.percentage_values, frozenset)


def test_evidence_extractor_rejects_non_evidence_input():
    with pytest.raises(TypeError):
        extract_evidence_numeric_values({})


def test_answer_number_extractor_rejects_non_string_input():
    with pytest.raises(TypeError):
        extract_answer_numbers(123)


def test_validator_accepts_supported_absolute_values():
    result = AnswerValidator().validate(
        "Revenue was 120.00 and the variance was -15.",
        _evidence(),
    )

    assert result.valid is True
    assert result.issues == ()


def test_validator_accepts_supported_percentages():
    result = AnswerValidator().validate(
        "Revenue increased by 20.0% and variance was -15%.",
        _evidence(),
    )

    assert result.valid is True
    assert result.issues == ()


def test_validator_flags_unsupported_absolute_value():
    result = AnswerValidator().validate(
        "Revenue was 999.",
        _evidence(),
    )

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].code == "unsupported_number"
    assert result.issues[0].offending_text == "999"


def test_validator_flags_unsupported_percentage():
    result = AnswerValidator().validate(
        "Revenue increased by 99%.",
        _evidence(),
    )

    assert result.valid is False
    assert len(result.issues) == 1
    assert result.issues[0].code == "unsupported_percentage"


def test_validator_does_not_confuse_absolute_and_percentage_values():
    result = AnswerValidator().validate(
        "Revenue increased by 120%.",
        _evidence(),
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_percentage"
        for issue in result.issues
    )


def test_validator_deduplicates_repeated_unsupported_values():
    result = AnswerValidator().validate(
        "Revenue was 999. The total was also 999.",
        _evidence(),
    )

    number_issues = [
        issue for issue in result.issues
        if issue.code == "unsupported_number"
    ]

    assert len(number_issues) == 1


def test_validator_accepts_matching_inr_notation():
    result = AnswerValidator().validate(
        "Revenue was ₹120.",
        _evidence(currency="INR"),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


def test_validator_flags_usd_notation_for_inr_evidence():
    result = AnswerValidator().validate(
        "Revenue was US$120.",
        _evidence(currency="INR"),
    )

    assert any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


def test_validator_accepts_usd_notation_for_usd_evidence():
    result = AnswerValidator().validate(
        "Revenue was US$120.",
        _evidence(currency="USD"),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


def test_validator_flags_mixed_currency_notation():
    result = AnswerValidator().validate(
        "Revenue was ₹120 and US$120.",
        _evidence(currency="INR"),
    )

    currency_issues = [
        issue for issue in result.issues
        if issue.code == "currency_mismatch"
    ]

    assert len(currency_issues) == 1
    assert currency_issues[0].offending_text == "USD"


def test_validator_does_not_flag_answers_without_currency_notation():
    result = AnswerValidator().validate(
        "Revenue was 120.",
        _evidence(currency="INR"),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


def test_validator_rejects_non_string_answer():
    with pytest.raises(TypeError):
        AnswerValidator().validate(123, _evidence())


def test_validator_rejects_invalid_evidence():
    with pytest.raises(TypeError):
        AnswerValidator().validate("Revenue was 120.", {})


def test_validator_flags_direction_contradiction():
    dataframe = pd.DataFrame({"revenue": [100.0, 120.0]})

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        trend=TrendResult(
            period_column="month",
            value_column="revenue",
            points=(
                TrendPoint("Jan", 100.0),
                TrendPoint("Feb", 120.0),
            ),
            first_value=100.0,
            last_value=120.0,
            absolute_change=20.0,
            percentage_change=20.0,
            direction="increasing",
        ),
    )

    result = AnswerValidator().validate(
        "Revenue decreased by 20%.",
        evidence,
    )

    assert any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


def test_validator_accepts_matching_increasing_direction():
    result = AnswerValidator().validate(
        "Revenue increased by 20%.",
        _evidence(),
    )

    assert not any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


def test_validator_skips_direction_check_when_evidence_conflicts():
    dataframe = pd.DataFrame({"revenue": [100.0, 120.0]})

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        trend=TrendResult(
            period_column="month",
            value_column="revenue",
            points=(
                TrendPoint("Jan", 100.0),
                TrendPoint("Feb", 120.0),
            ),
            first_value=100.0,
            last_value=120.0,
            absolute_change=20.0,
            percentage_change=20.0,
            direction="increasing",
        ),
        variance=VarianceResult(
            actual_value=80.0,
            reference_value=100.0,
            absolute_variance=-20.0,
            percentage_variance=-20.0,
            direction="below_reference",
        ),
    )

    result = AnswerValidator().validate(
        "Revenue decreased by 20%.",
        evidence,
    )

    assert not any(
        issue.code == "direction_mismatch"
        for issue in result.issues
    )


def test_validator_flags_causal_language_with_root_cause_evidence():
    from datapilot.analytics.contribution import (
        Contribution,
        ContributionResult,
    )
    from datapilot.analytics.root_cause import RootCauseAnalyzer

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

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        root_cause=root_cause,
    )

    result = AnswerValidator().validate(
        "The North region caused the increase.",
        evidence,
    )

    assert any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )


def test_validator_allows_causal_wording_without_root_cause_evidence():
    result = AnswerValidator().validate(
        "The increase was caused by higher revenue.",
        _evidence(),
    )

    assert not any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )


def test_validator_flags_multiple_causal_phrases():
    from datapilot.analytics.contribution import (
        Contribution,
        ContributionResult,
    )
    from datapilot.analytics.root_cause import RootCauseAnalyzer

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

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        root_cause=root_cause,
    )

    result = AnswerValidator().validate(
        "North caused the increase, resulting in higher revenue.",
        evidence,
    )

    causal_issues = [
        issue for issue in result.issues
        if issue.code == "unsupported_causal_claim"
    ]

    assert len(causal_issues) == 2


def test_validator_does_not_flag_contribution_driver_description():
    from datapilot.analytics.contribution import (
        Contribution,
        ContributionResult,
    )
    from datapilot.analytics.root_cause import RootCauseAnalyzer

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

    evidence = EvidenceBuilder().build(
        dataframe=dataframe,
        root_cause=root_cause,
    )

    result = AnswerValidator().validate(
        "North contributed 20 to the overall change.",
        evidence,
    )

    assert not any(
        issue.code == "unsupported_causal_claim"
        for issue in result.issues
    )
