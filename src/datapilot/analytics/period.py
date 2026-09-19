from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodComparison:
    """Deterministic comparison between two analytical periods."""

    value_a: float
    value_b: float
    absolute_change: float
    percentage_change: float | None
    direction: str


class PeriodComparator:
    """Compare two analytical period values deterministically."""

    def compare(
        self,
        value_a: float,
        value_b: float,
    ) -> PeriodComparison:
        """Compare an earlier/reference value with a later/current value."""

        value_a = float(value_a)
        value_b = float(value_b)

        absolute_change = value_b - value_a

        if value_a == 0:
            percentage_change = None
        else:
            percentage_change = (
                absolute_change
                / abs(value_a)
            ) * 100

        if absolute_change > 0:
            direction = "increased"
        elif absolute_change < 0:
            direction = "decreased"
        else:
            direction = "unchanged"

        return PeriodComparison(
            value_a=value_a,
            value_b=value_b,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            direction=direction,
        )