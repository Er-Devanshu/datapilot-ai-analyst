from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.analyzer import (
    AnalysisResult,
    CategorySummary,
    NumericSummary,
)
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.trend import TrendResult


@dataclass(frozen=True)
class Evidence:
    """Structured, grounded evidence for answer composition."""

    row_count: int
    numeric_summaries: tuple[NumericSummary, ...]
    category_summaries: tuple[CategorySummary, ...]
    result_columns: tuple[str, ...]
    currency: str = "INR"
    trend: TrendResult | None = None
    period_comparison: PeriodComparison | None = None

    def to_prompt_text(self) -> str:
        """Convert evidence into deterministic prompt context."""

        lines = [
            "EVIDENCE:",
            f"Result rows: {self.row_count:,}",
            f"Currency: {self.currency}",
            "",
            "NUMERIC SUMMARIES:",
        ]

        if self.numeric_summaries:
            for summary in self.numeric_summaries:
                lines.extend(
                    [
                        f"- Column: {summary.column}",
                        f"  Total: {summary.total}",
                        f"  Average: {summary.average}",
                        f"  Minimum: {summary.minimum}",
                        f"  Maximum: {summary.maximum}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "CATEGORY SUMMARIES:",
            ]
        )

        if self.category_summaries:
            for summary in self.category_summaries:
                lines.extend(
                    [
                        f"- Dimension: {summary.dimension}",
                        f"  Measure: {summary.measure}",
                        f"  Top category: {summary.top_category}",
                        f"  Top value: {summary.top_value}",
                        f"  Bottom category: {summary.bottom_category}",
                        f"  Bottom value: {summary.bottom_value}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "TREND ANALYSIS:",
            ]
        )

        if self.trend is not None:
            lines.extend(
                [
                    f"- Period column: {self.trend.period_column}",
                    f"- Value column: {self.trend.value_column}",
                    f"- First value: {self.trend.first_value}",
                    f"- Last value: {self.trend.last_value}",
                    f"- Absolute change: {self.trend.absolute_change}",
                    f"- Percentage change: {self.trend.percentage_change}",
                    f"- Direction: {self.trend.direction}",
                ]
            )

            lines.append("- Trend points:")

            for point in self.trend.points:
                lines.append(
                    f"  - {point.period}: {point.value}"
                )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "PERIOD COMPARISON:",
            ]
        )

        if self.period_comparison is not None:
            comparison = self.period_comparison

            lines.extend(
                [
                    f"- Reference value: {comparison.value_a}",
                    f"- Current value: {comparison.value_b}",
                    f"- Absolute change: {comparison.absolute_change}",
                    f"- Percentage change: {comparison.percentage_change}",
                    f"- Direction: {comparison.direction}",
                ]
            )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "RESULT COLUMNS:",
                f"- {', '.join(self.result_columns)}",
            ]
        )

        return "\n".join(lines)


class EvidenceBuilder:
    """Build grounded evidence from deterministic analysis."""

    def __init__(
        self,
        currency: str = "INR",
    ) -> None:
        if not currency.strip():
            raise ValueError(
                "Currency cannot be empty."
            )

        self.currency = currency.strip()

    def build(
        self,
        dataframe: pd.DataFrame,
        analysis: AnalysisResult | None = None,
        trend: TrendResult | None = None,
        period_comparison: PeriodComparison | None = None,
    ) -> Evidence:
        """Build evidence from deterministic analytical results."""

        if analysis is None:
            from datapilot.analytics.analyzer import ResultAnalyzer

            analysis = ResultAnalyzer().analyze(
                dataframe
            )

        if analysis.row_count != len(dataframe):
            raise ValueError(
                "Analysis row count does not match "
                "the query result."
            )

        return Evidence(
            row_count=analysis.row_count,
            numeric_summaries=tuple(
                analysis.numeric_summaries
            ),
            category_summaries=tuple(
                analysis.category_summaries
            ),
            result_columns=tuple(
                str(column)
                for column in dataframe.columns
            ),
            currency=self.currency,
            trend=trend,
            period_comparison=period_comparison,
        )