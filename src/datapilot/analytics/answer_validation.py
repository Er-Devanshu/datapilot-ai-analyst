from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from datapilot.analytics.evidence import Evidence
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.root_cause import RootCauseResult
from datapilot.analytics.trend import TrendResult
from datapilot.analytics.variance import VarianceResult


@dataclass(frozen=True)
class AnswerValidationIssue:
    """One deterministic finding raised while validating an answer."""

    code: str
    message: str
    severity: str = "warning"
    offending_text: str | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("Validation issue code cannot be empty.")
        if not self.message.strip():
            raise ValueError("Validation issue message cannot be empty.")
        if self.severity not in {"warning", "error"}:
            raise ValueError("Severity must be warning or error.")


@dataclass(frozen=True)
class AnswerValidationResult:
    """Immutable outcome of deterministic answer validation."""

    valid: bool
    issues: tuple[AnswerValidationIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("Validation issues must be provided as a tuple.")
        if not all(
            isinstance(issue, AnswerValidationIssue)
            for issue in self.issues
        ):
            raise TypeError(
                "Validation issues must contain AnswerValidationIssue values."
            )
        if self.valid and self.issues:
            raise ValueError("A valid result cannot contain validation issues.")

    @classmethod
    def success(cls) -> AnswerValidationResult:
        return cls(valid=True, issues=())

    @classmethod
    def with_issues(
        cls,
        issues: tuple[AnswerValidationIssue, ...],
    ) -> AnswerValidationResult:
        return cls(valid=not issues, issues=issues)


@dataclass(frozen=True)
class EvidenceNumericValues:
    """Numeric evidence separated by semantic type."""

    absolute_values: frozenset[Decimal]
    percentage_values: frozenset[Decimal]


_PERCENT_FIELD_PATTERN = re.compile(r"percent|percentage", re.IGNORECASE)

_IGNORED_NUMERIC_FIELDS = {
    "row_count",
    "count",
    "request_id",
    "limit",
}

_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)

# Currency codes may be adjacent to digits (for example, USD120).
# Ambiguous standalone "$" and "¥" symbols are deliberately excluded.
_CURRENCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "INR",
        re.compile(r"₹|(?<![A-Za-z])INR(?![A-Za-z])|(?<![A-Za-z])Rs\.?(?![A-Za-z])|(?<![A-Za-z])rupees?(?![A-Za-z])", re.IGNORECASE),
    ),
    ("CAD", re.compile(r"(?<![A-Za-z])C\$(?![A-Za-z])|(?<![A-Za-z])CAD(?![A-Za-z])", re.IGNORECASE)),
    ("AUD", re.compile(r"(?<![A-Za-z])A\$(?![A-Za-z])|(?<![A-Za-z])AUD(?![A-Za-z])", re.IGNORECASE)),
    ("USD", re.compile(r"(?<![A-Za-z])US\$(?![A-Za-z])|(?<![A-Za-z])USD(?![A-Za-z])", re.IGNORECASE)),
    ("EUR", re.compile(r"€|(?<![A-Za-z])EUR(?![A-Za-z])", re.IGNORECASE)),
    ("GBP", re.compile(r"£|(?<![A-Za-z])GBP(?![A-Za-z])", re.IGNORECASE)),
    ("JPY", re.compile(r"(?<![A-Za-z])JPY(?![A-Za-z])", re.IGNORECASE)),
)

_DIRECTION_GROUPS = {
    "positive": (
        re.compile(
            r"\b(?:increas(?:e|ed|es|ing)|ris(?:e|es|en|ing)|"
            r"grew|growth|up|higher|above|positive)\b",
            re.IGNORECASE,
        ),
    ),
    "negative": (
        re.compile(
            r"\b(?:decreas(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|"
            r"fell|fall(?:s|en|ing)?|drop(?:ped|s|ping)?|"
            r"down|lower|below|negative)\b",
            re.IGNORECASE,
        ),
    ),
    "neutral": (
        re.compile(
            r"\b(?:stable|unchanged|no change|at reference|"
            r"remained constant)\b",
            re.IGNORECASE,
        ),
    ),
}

_CAUSAL_PATTERNS = (
    re.compile(r"\bcaused by\b", re.IGNORECASE),
    re.compile(r"\bcaused\b", re.IGNORECASE),
    re.compile(r"\b(?:led|leads|leading) to\b", re.IGNORECASE),
    re.compile(r"\bresulted in\b", re.IGNORECASE),
    re.compile(r"\bresulting in\b", re.IGNORECASE),
    re.compile(r"\bbecause of\b", re.IGNORECASE),
    re.compile(r"\bdue to\b", re.IGNORECASE),
    re.compile(r"\b(?:the )?root cause\b", re.IGNORECASE),
    re.compile(r"\b(?:was|were) responsible for\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are) responsible for\b", re.IGNORECASE),
)

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|"
    r"didn't|doesn't|don't|"
    r"wasn't|weren't|isn't|aren't|"
    r"hasn't|haven't|hadn't|"
    r"cannot|can't|couldn't|wouldn't|shouldn't)"
    r"(?:\s+\w+){0,2}\s*$",
    re.IGNORECASE,
)

_QUOTE_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("‘", "’"),
)


def _mask_quoted_text(text: str) -> str:
    """Mask quoted spans while preserving character positions."""

    characters = list(text)

    for opening, closing in _QUOTE_PAIRS:
        index = 0

        while index < len(text):
            start = text.find(opening, index)
            if start == -1:
                break

            end = text.find(closing, start + len(opening))
            if end == -1:
                break

            for position in range(start, end + len(closing)):
                characters[position] = " "

            index = end + len(closing)

    return "".join(characters)


def _is_negated_claim(text: str, match_start: int) -> bool:
    """Check for a nearby negation preceding a matched claim."""

    prefix = text[max(0, match_start - 60):match_start]
    return _NEGATION_PATTERN.search(prefix) is not None


def _to_decimal(value: Any) -> Decimal | None:
    """Convert supported numeric values to finite Decimal values."""

    if isinstance(value, bool):
        return None

    if not isinstance(value, (int, float, Decimal)):
        return None

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

    if not result.is_finite():
        return None

    return result


def _collect_numeric_fields(
    value: Any,
    absolute_values: set[Decimal],
    percentage_values: set[Decimal],
    field_name: str = "",
) -> None:
    """Recursively collect numeric dataclass fields by semantic name."""

    if field_name.casefold() in _IGNORED_NUMERIC_FIELDS:
        return

    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _collect_numeric_fields(
                getattr(value, item.name),
                absolute_values,
                percentage_values,
                item.name,
            )
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_numeric_fields(
                item,
                absolute_values,
                percentage_values,
                field_name,
            )
        return

    number = _to_decimal(value)
    if number is None:
        return

    if _PERCENT_FIELD_PATTERN.search(field_name):
        percentage_values.add(number)
    else:
        absolute_values.add(number)


def extract_evidence_numeric_values(
    evidence: Evidence,
) -> EvidenceNumericValues:
    """Extract typed numeric values from structured evidence."""

    if not isinstance(evidence, Evidence):
        raise TypeError("Evidence must be an Evidence instance.")

    absolute_values: set[Decimal] = set()
    percentage_values: set[Decimal] = set()

    sources = (
        evidence.numeric_summaries,
        evidence.category_summaries,
        evidence.analytics_executions,
        evidence.trend,
        evidence.period_comparison,
        evidence.variance,
        evidence.contribution,
        evidence.segmentation,
        evidence.root_cause,
    )

    for source in sources:
        _collect_numeric_fields(
            source,
            absolute_values,
            percentage_values,
        )

    return EvidenceNumericValues(
        absolute_values=frozenset(absolute_values),
        percentage_values=frozenset(percentage_values),
    )


def extract_answer_numbers(answer: str) -> tuple[tuple[str, Decimal, bool], ...]:
    """Extract numeric mentions while excluding dates and identifiers."""

    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    date_spans = [
        match.span()
        for match in _DATE_PATTERN.finditer(answer)
    ]

    number_pattern = re.compile(
        r"(?<![\w.-])"
        r"[-+]?"
        r"(?:"
        r"\d{1,2}(?:,\d{2})*,\d{3}"
        r"|"
        r"\d{1,3}(?:,\d{3})+"
        r"|"
        r"\d+"
        r")"
        r"(?:\.\d+)?"
        r"(?![\w,])"
        r"(?:\s*%)?"
    )

    extracted: list[tuple[str, Decimal, bool]] = []

    for match in number_pattern.finditer(answer):
        start, end = match.span()

        if any(
            start < date_end and end > date_start
            for date_start, date_end in date_spans
        ):
            continue

        original = match.group().strip()
        is_percentage = original.endswith("%")
        normalized = original.removesuffix("%").strip().replace(",", "")

        try:
            number = Decimal(normalized)
        except InvalidOperation:
            continue

        if not number.is_finite():
            continue

        extracted.append((original, number, is_percentage))

    return tuple(extracted)


def _normalize_currency(currency: str) -> str:
    """Normalize supported currency codes and common names."""

    normalized = currency.strip().upper()

    aliases = {
        "₹": "INR",
        "RS": "INR",
        "RUPEE": "INR",
        "RUPEES": "INR",
        "$": "USD",
        "US$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "C$": "CAD",
        "A$": "AUD",
    }

    return aliases.get(normalized, normalized)


def _extract_currency_codes(answer: str) -> frozenset[str]:
    """Identify only unambiguous supported currency notation."""

    return frozenset(
        currency
        for currency, pattern in _CURRENCY_PATTERNS
        if pattern.search(answer)
    )


def _direction_from_value(direction: str) -> str | None:
    """Normalize a structured analytics direction."""

    normalized = direction.casefold().strip()

    if normalized in {
        "increasing",
        "increased",
        "above_reference",
        "positive",
    }:
        return "positive"

    if normalized in {
        "decreasing",
        "decreased",
        "below_reference",
        "negative",
    }:
        return "negative"

    if normalized in {
        "stable",
        "unchanged",
        "at_reference",
        "neutral",
    }:
        return "neutral"

    return None


def _get_directional_evidence(evidence: Evidence) -> set[str]:
    """Collect recognized directions from structured analytics results."""

    directions: set[str] = set()

    sources: list[Any] = [
        evidence.trend,
        evidence.period_comparison,
        evidence.variance,
        evidence.root_cause,
    ]

    sources.extend(
        execution.result
        for execution in evidence.analytics_executions
    )

    for source in sources:
        direction: str | None = None

        if isinstance(source, TrendResult):
            direction = source.direction
        elif isinstance(source, PeriodComparison):
            direction = source.direction
        elif isinstance(source, VarianceResult):
            direction = source.direction
        elif isinstance(source, RootCauseResult):
            direction = source.overall_direction

        if direction is None:
            continue

        normalized = _direction_from_value(direction)
        if normalized is not None:
            directions.add(normalized)

    return directions


def _is_root_cause_evidence_present(evidence: Evidence) -> bool:
    """Check whether root-cause contribution decomposition is available."""

    if evidence.root_cause is not None:
        return True

    return any(
        isinstance(execution.result, RootCauseResult)
        for execution in evidence.analytics_executions
    )


class AnswerValidator:
    """Validate numerical, directional, currency, and causal claims."""

    def validate(
        self,
        answer: str,
        evidence: Evidence,
    ) -> AnswerValidationResult:
        """Return non-blocking findings without modifying the answer."""

        if not isinstance(answer, str):
            raise TypeError("Answer must be a string.")

        if not isinstance(evidence, Evidence):
            raise TypeError("Evidence must be an Evidence instance.")

        if not answer.strip():
            return AnswerValidationResult.success()

        issues: list[AnswerValidationIssue] = []
        numeric_evidence = extract_evidence_numeric_values(evidence)
        seen_numeric: set[tuple[str, str]] = set()

        for original, number, is_percentage in extract_answer_numbers(answer):
            evidence_values = (
                numeric_evidence.percentage_values
                if is_percentage
                else numeric_evidence.absolute_values
            )

            if number in evidence_values:
                continue

            code = (
                "unsupported_percentage"
                if is_percentage
                else "unsupported_number"
            )
            semantic_type = "percentage" if is_percentage else "absolute value"
            issue_key = (code, original)

            if issue_key in seen_numeric:
                continue

            seen_numeric.add(issue_key)
            issues.append(
                AnswerValidationIssue(
                    code=code,
                    message=(
                        f"The {semantic_type} {original} was not found "
                        "in the structured evidence."
                    ),
                    offending_text=original,
                )
            )

        expected_currency = _normalize_currency(evidence.currency)
        answer_currencies = _extract_currency_codes(answer)

        for currency in sorted(answer_currencies):
            if currency == expected_currency:
                continue

            issues.append(
                AnswerValidationIssue(
                    code="currency_mismatch",
                    message=(
                        f"The answer uses {currency} notation, but the "
                        f"evidence currency is {expected_currency}."
                    ),
                    offending_text=currency,
                )
            )

        unquoted_answer = _mask_quoted_text(answer)
        directional_evidence = _get_directional_evidence(evidence)

        if len(directional_evidence) == 1:
            expected_direction = next(iter(directional_evidence))
            opposite_directions = {
                "positive": {"negative"},
                "negative": {"positive"},
                "neutral": {"positive", "negative"},
            }[expected_direction]

            for claimed_direction, patterns in _DIRECTION_GROUPS.items():
                if claimed_direction not in opposite_directions:
                    continue

                for pattern in patterns:
                    for match in pattern.finditer(unquoted_answer):
                        if _is_negated_claim(unquoted_answer, match.start()):
                            continue

                        issues.append(
                            AnswerValidationIssue(
                                code="direction_mismatch",
                                message=(
                                    "The answer's directional wording may "
                                    "contradict the unambiguous direction "
                                    "reported by structured analytics evidence."
                                ),
                                offending_text=match.group(),
                            )
                        )
                        break

        if _is_root_cause_evidence_present(evidence):
            for pattern in _CAUSAL_PATTERNS:
                for match in pattern.finditer(unquoted_answer):
                    if _is_negated_claim(unquoted_answer, match.start()):
                        continue

                    issues.append(
                        AnswerValidationIssue(
                            code="unsupported_causal_claim",
                            message=(
                                "Root-cause evidence provides a deterministic "
                                "contribution decomposition, not proof of "
                                "causation. Review this causal wording."
                            ),
                            offending_text=match.group(),
                        )
                    )

        return AnswerValidationResult.with_issues(tuple(issues))