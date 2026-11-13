
from decimal import Decimal

import pandas as pd
import pytest

from datapilot.analytics.answer_validation import (
    AnswerValidator,
    extract_answer_numbers,
)
from datapilot.analytics.evidence import EvidenceBuilder


def _evidence(currency: str):
    return EvidenceBuilder(currency=currency).build(
        pd.DataFrame({"revenue": [120.0]})
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Revenue was ₹1,25,000.",
            (("1,25,000", Decimal("125000"), False),),
        ),
        (
            "Revenue was 12,34,567.",
            (("12,34,567", Decimal("1234567"), False),),
        ),
        (
            "Revenue was 1,00,000.50.",
            (("1,00,000.50", Decimal("100000.50"), False),),
        ),
        (
            "Revenue was 1,234,567.89.",
            (("1,234,567.89", Decimal("1234567.89"), False),),
        ),
        (
            "Revenue was 1,25,000 and growth was 12.5%.",
            (
                ("1,25,000", Decimal("125000"), False),
                ("12.5%", Decimal("12.5"), True),
            ),
        ),
    ],
)
def test_extract_answer_numbers_supports_indian_and_international_grouping(
    text: str,
    expected: tuple[tuple[str, Decimal, bool], ...],
) -> None:
    assert extract_answer_numbers(text) == expected


@pytest.mark.parametrize(
    ("currency", "notation"),
    [
        ("CAD", "C$"),
        ("AUD", "A$"),
        ("USD", "US$"),
        ("USD", "USD"),
        ("JPY", "JPY"),
        ("INR", "₹"),
        ("EUR", "€"),
        ("GBP", "£"),
    ],
)
def test_explicit_currency_notation_matches_evidence(
    currency: str,
    notation: str,
) -> None:
    result = AnswerValidator().validate(
        f"Revenue was {notation}120.",
        _evidence(currency),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    ("answer_notation", "expected_currency"),
    [
        ("C$", "CAD"),
        ("A$", "AUD"),
        ("US$", "USD"),
        ("USD", "USD"),
        ("EUR", "EUR"),
        ("GBP", "GBP"),
        ("JPY", "JPY"),
    ],
)
def test_explicit_foreign_currency_notation_reports_mismatch(
    answer_notation: str,
    expected_currency: str,
) -> None:
    result = AnswerValidator().validate(
        f"Revenue was {answer_notation}120.",
        _evidence("INR"),
    )

    currency_issues = [
        issue
        for issue in result.issues
        if issue.code == "currency_mismatch"
    ]

    assert len(currency_issues) == 1
    assert currency_issues[0].offending_text == expected_currency


@pytest.mark.parametrize(
    "evidence_currency",
    ["CAD", "AUD", "USD", "INR"],
)
def test_standalone_dollar_symbol_is_not_assumed_to_be_usd(
    evidence_currency: str,
) -> None:
    result = AnswerValidator().validate(
        "Revenue was $120.",
        _evidence(evidence_currency),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "evidence_currency",
    ["JPY", "CNY"],
)
def test_standalone_yen_symbol_is_not_assumed_to_be_jpy(
    evidence_currency: str,
) -> None:
    result = AnswerValidator().validate(
        "Revenue was ¥120.",
        _evidence(evidence_currency),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )


def test_mixed_explicit_currency_notation_reports_only_foreign_currency():
    result = AnswerValidator().validate(
        "Revenue was ₹120 and US$120.",
        _evidence("INR"),
    )

    currency_issues = [
        issue
        for issue in result.issues
        if issue.code == "currency_mismatch"
    ]

    assert len(currency_issues) == 1
    assert currency_issues[0].offending_text == "USD"


def test_answer_without_currency_notation_has_no_currency_warning():
    result = AnswerValidator().validate(
        "Revenue was 120.",
        _evidence("INR"),
    )

    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.issues
    )
