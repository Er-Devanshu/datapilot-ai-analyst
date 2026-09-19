import pytest

from datapilot.analytics.period import PeriodComparator


def test_period_comparison_detects_increase() -> None:
    result = PeriodComparator().compare(
        value_a=100.0,
        value_b=125.0,
    )

    assert result.value_a == 100.0
    assert result.value_b == 125.0
    assert result.absolute_change == 25.0
    assert result.percentage_change == 25.0
    assert result.direction == "increased"


def test_period_comparison_detects_decrease() -> None:
    result = PeriodComparator().compare(
        value_a=200.0,
        value_b=150.0,
    )

    assert result.value_a == 200.0
    assert result.value_b == 150.0
    assert result.absolute_change == -50.0
    assert result.percentage_change == -25.0
    assert result.direction == "decreased"


def test_period_comparison_detects_unchanged_value() -> None:
    result = PeriodComparator().compare(
        value_a=100.0,
        value_b=100.0,
    )

    assert result.absolute_change == 0.0
    assert result.percentage_change == 0.0
    assert result.direction == "unchanged"


def test_period_comparison_handles_zero_reference() -> None:
    result = PeriodComparator().compare(
        value_a=0.0,
        value_b=100.0,
    )

    assert result.absolute_change == 100.0
    assert result.percentage_change is None
    assert result.direction == "increased"


def test_period_comparison_handles_zero_current_value() -> None:
    result = PeriodComparator().compare(
        value_a=100.0,
        value_b=0.0,
    )

    assert result.absolute_change == -100.0
    assert result.percentage_change == -100.0
    assert result.direction == "decreased"


def test_period_comparison_handles_negative_reference() -> None:
    result = PeriodComparator().compare(
        value_a=-100.0,
        value_b=-50.0,
    )

    assert result.absolute_change == 50.0
    assert result.percentage_change == 50.0
    assert result.direction == "increased"


def test_period_comparison_converts_numeric_values() -> None:
    result = PeriodComparator().compare(
        value_a=100,
        value_b=150,
    )

    assert isinstance(result.value_a, float)
    assert isinstance(result.value_b, float)
    assert result.absolute_change == 50.0
    assert result.percentage_change == 50.0