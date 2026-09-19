from __future__ import annotations

import pytest

from datapilot.analytics.variance import VarianceAnalyzer


def test_variance_above_reference() -> None:
    result = VarianceAnalyzer().analyze(
        actual_value=1200,
        reference_value=1000,
    )

    assert result.actual_value == 1200
    assert result.reference_value == 1000
    assert result.absolute_variance == 200
    assert result.percentage_variance == 20
    assert result.direction == "above_reference"


def test_variance_below_reference() -> None:
    result = VarianceAnalyzer().analyze(
        actual_value=800,
        reference_value=1000,
    )

    assert result.absolute_variance == -200
    assert result.percentage_variance == -20
    assert result.direction == "below_reference"


def test_variance_at_reference() -> None:
    result = VarianceAnalyzer().analyze(
        actual_value=1000,
        reference_value=1000,
    )

    assert result.absolute_variance == 0
    assert result.percentage_variance == 0
    assert result.direction == "at_reference"


def test_variance_with_zero_reference() -> None:
    result = VarianceAnalyzer().analyze(
        actual_value=1000,
        reference_value=0,
    )

    assert result.absolute_variance == 1000
    assert result.percentage_variance is None
    assert result.direction == "above_reference"


def test_negative_reference_uses_absolute_denominator() -> None:
    result = VarianceAnalyzer().analyze(
        actual_value=-80,
        reference_value=-100,
    )

    assert result.absolute_variance == 20
    assert result.percentage_variance == 20
    assert result.direction == "above_reference"