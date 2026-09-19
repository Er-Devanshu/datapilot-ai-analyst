from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.analyzer import (
    AnalysisResult,
    CategorySummary,
    NumericSummary,
)


@dataclass(frozen=True)
class Evidence:
    """Structured, grounded evidence for answer composition."""

    row_count: int
    numeric_summaries: tuple[NumericSummary, ...]
    category_summaries: tuple[CategorySummary, ...]
    result_columns: tuple[str, ...]
    currency: str = "INR"

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
                "RESULT COLUMNS:",
                f"- {', '.join(self.result_columns)}",
            ]
        )

        return "\n".join(lines)


class EvidenceBuilder:
    """Build grounded evidence from deterministic query analysis."""

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
    ) -> Evidence:
        """Build evidence from a query result and its analysis."""

        if analysis is None:
            from datapilot.analytics.analyzer import ResultAnalyzer

            analysis = ResultAnalyzer().analyze(dataframe)

        if analysis.row_count != len(dataframe):
            raise ValueError(
                "Analysis row count does not match the query result."
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
        )