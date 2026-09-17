from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def to_prompt_text(self) -> str:
        """Convert evidence into deterministic prompt context."""

        lines = [
            "EVIDENCE:",
            f"Result rows: {self.row_count:,}",
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
                lines.append(
                    f"- Category column: {summary.category_column}"
                )
                lines.append(
                    f"  Measure column: {summary.measure_column}"
                )
                lines.append(
                    f"  Top categories: {summary.top_categories}"
                )
                lines.append(
                    f"  Bottom categories: {summary.bottom_categories}"
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
        )