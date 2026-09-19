from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VarianceResult:
    """Deterministic variance between an actual and reference value."""

    actual_value: float
    reference_value: float
    absolute_variance: float
    percentage_variance: float | None
    direction: str


class VarianceAnalyzer:
    """Calculate deterministic variance between two values."""

    def analyze(
        self,
        actual_value: float,
        reference_value: float,
    ) -> VarianceResult:
        """Calculate absolute and percentage variance."""

        actual_value = float(actual_value)
        reference_value = float(reference_value)

        absolute_variance = (
            actual_value - reference_value
        )

        if reference_value == 0:
            percentage_variance = None
        else:
            percentage_variance = (
                absolute_variance
                / abs(reference_value)
            ) * 100

        if absolute_variance > 0:
            direction = "above_reference"
        elif absolute_variance < 0:
            direction = "below_reference"
        else:
            direction = "at_reference"

        return VarianceResult(
            actual_value=actual_value,
            reference_value=reference_value,
            absolute_variance=absolute_variance,
            percentage_variance=percentage_variance,
            direction=direction,
        )