
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.analyzer import (
    AnalysisResult,
    CategorySummary,
    NumericSummary,
)
from datapilot.analytics.contribution import ContributionResult
from datapilot.analytics.executor import AnalysisExecution
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.root_cause import RootCauseResult
from datapilot.analytics.segmentation import SegmentationResult
from datapilot.analytics.trend import TrendResult
from datapilot.analytics.variance import VarianceResult
from datapilot.visualization.models import VisualizationSpec


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
    variance: VarianceResult | None = None
    contribution: ContributionResult | None = None
    segmentation: SegmentationResult | None = None
    root_cause: RootCauseResult | None = None
    visualization: VisualizationSpec | None = None
    analytics_executions: tuple[AnalysisExecution, ...] = ()

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

        lines.extend(["", "CATEGORY SUMMARIES:"])

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

        lines.extend(["", "TREND ANALYSIS:"])

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
                    "- Trend points:",
                ]
            )

            for point in self.trend.points:
                lines.append(
                    f"  - {point.period}: {point.value}"
                )
        else:
            lines.append("- None")

        lines.extend(["", "PERIOD COMPARISON:"])

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

        lines.extend(["", "VARIANCE ANALYSIS:"])

        if self.variance is not None:
            variance = self.variance
            lines.extend(
                [
                    f"- Actual value: {variance.actual_value}",
                    f"- Reference value: {variance.reference_value}",
                    f"- Absolute variance: {variance.absolute_variance}",
                    f"- Percentage variance: {variance.percentage_variance}",
                    f"- Direction: {variance.direction}",
                ]
            )
        else:
            lines.append("- None")

        lines.extend(["", "CONTRIBUTION ANALYSIS:"])

        if self.contribution is not None:
            contribution = self.contribution
            lines.extend(
                [
                    f"- Dimension column: {contribution.dimension_column}",
                    f"- Value column: {contribution.value_column}",
                    f"- Total change: {contribution.total_change}",
                    "- Contributions:",
                ]
            )

            for item in contribution.contributions:
                lines.extend(
                    [
                        f"  - Category: {item.category}",
                        f"    Value: {item.value}",
                        "    Contribution percentage: "
                        f"{item.contribution_percentage}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(["", "SEGMENTATION ANALYSIS:"])

        if self.segmentation is not None:
            segmentation = self.segmentation
            lines.extend(
                [
                    f"- Dimension column: {segmentation.dimension_column}",
                    f"- Value column: {segmentation.value_column}",
                    f"- Total value: {segmentation.total_value}",
                    "- Segments:",
                ]
            )

            for segment in segmentation.segments:
                lines.extend(
                    [
                        f"  - Segment: {segment.segment}",
                        f"    Row count: {segment.row_count}",
                        f"    Total value: {segment.total_value}",
                        f"    Average value: {segment.average_value}",
                        f"    Minimum value: {segment.minimum_value}",
                        f"    Maximum value: {segment.maximum_value}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(["", "ROOT-CAUSE ANALYSIS:"])

        if self.root_cause is not None:
            root_cause = self.root_cause
            lines.extend(
                [
                    f"- Dimension column: {root_cause.dimension_column}",
                    f"- Value column: {root_cause.value_column}",
                    f"- Overall change: {root_cause.overall_change}",
                    f"- Overall direction: {root_cause.overall_direction}",
                    "- Drivers:",
                ]
            )

            for driver in root_cause.drivers:
                lines.extend(
                    [
                        f"  - Category: {driver.category}",
                        f"    Value: {driver.value}",
                        "    Contribution percentage: "
                        f"{driver.contribution_percentage}",
                        f"    Direction: {driver.direction}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(["", "ANALYTICS EXECUTION RESULTS:"])

        if self.analytics_executions:
            for execution in self.analytics_executions:
                lines.extend(
                    [
                        f"- Request ID: {execution.request_id}",
                        f"  Analysis type: {execution.analysis_type.value}",
                        f"  Deterministic result: {execution.result!r}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(["", "VISUALIZATION:"])

        if self.visualization is not None:
            visualization = self.visualization
            lines.extend(
                [
                    f"- Chart type: {visualization.chart_type}",
                    f"- X column: {visualization.x_column}",
                    f"- Y column: {visualization.y_column}",
                    f"- Color column: {visualization.color_column}",
                    f"- Title: {visualization.title}",
                    f"- Orientation: {visualization.orientation}",
                    f"- Sort by: {visualization.sort_by}",
                    f"- Sort descending: {visualization.sort_descending}",
                    f"- Limit: {visualization.limit}",
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
        variance: VarianceResult | None = None,
        contribution: ContributionResult | None = None,
        segmentation: SegmentationResult | None = None,
        root_cause: RootCauseResult | None = None,
        visualization: VisualizationSpec | None = None,
        analytics_executions: tuple[AnalysisExecution, ...] = (),
    ) -> Evidence:
        """Build evidence from deterministic analytical results."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Dataframe must be a pandas DataFrame."
            )

        if not isinstance(analytics_executions, tuple):
            raise TypeError(
                "Analytics executions must be provided as a tuple."
            )

        if not all(
            isinstance(item, AnalysisExecution)
            for item in analytics_executions
        ):
            raise TypeError(
                "Analytics executions must contain AnalysisExecution values."
            )

        if analysis is None:
            from datapilot.analytics.analyzer import ResultAnalyzer

            analysis = ResultAnalyzer().analyze(dataframe)

        if analysis.row_count != len(dataframe):
            raise ValueError(
                "Analysis row count does not match the query result."
            )

        return Evidence(
            row_count=analysis.row_count,
            numeric_summaries=tuple(analysis.numeric_summaries),
            category_summaries=tuple(analysis.category_summaries),
            result_columns=tuple(
                str(column)
                for column in dataframe.columns
            ),
            currency=self.currency,
            trend=trend,
            period_comparison=period_comparison,
            variance=variance,
            contribution=contribution,
            segmentation=segmentation,
            root_cause=root_cause,
            visualization=visualization,
            analytics_executions=analytics_executions,
        )
