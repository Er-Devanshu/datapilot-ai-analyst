
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.analyzer import (
    AnalysisResult,
    CategorySummary,
    NumericSummary,
)
from datapilot.analytics.contribution import ContributionResult
from datapilot.analytics.executor import (
    AnalysisExecution,
    AnalyticsExecutionResult,
)
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.planning import (
    PlanValidationIssue,
    SkippedAnalysis,
)
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
    visualization_source: str | None = None
    visualization_request_id: str | None = None
    skipped_analyses: tuple[SkippedAnalysis, ...] = ()
    fatal_errors: tuple[PlanValidationIssue, ...] = ()

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

        self._append_legacy_analysis(lines)

        lines.extend(["", "ANALYTICS EXECUTION RESULTS:"])

        if self.analytics_executions:
            for execution in self.analytics_executions:
                lines.extend(
                    [
                        f"- Request ID: {execution.request_id}",
                        f"  Analysis type: {execution.analysis_type.value}",
                    ]
                )
                lines.extend(
                    self._format_analysis_result(execution.result)
                )
        else:
            lines.append("- None")

        lines.extend(["", "SKIPPED ANALYSES:"])

        if self.skipped_analyses:
            for skipped in self.skipped_analyses:
                lines.extend(
                    [
                        f"- Request ID: {skipped.request_id}",
                        f"  Analysis type: {skipped.analysis_type.value}",
                        f"  Reason: {skipped.reason}",
                    ]
                )
        else:
            lines.append("- None")

        lines.extend(["", "FATAL ANALYTICS ERRORS:"])

        if self.fatal_errors:
            for issue in self.fatal_errors:
                lines.extend(
                    [
                        f"- Code: {issue.code}",
                        f"  Message: {issue.message}",
                        f"  Request ID: {issue.request_id}",
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
                    f"- Visualization source: {self.visualization_source}",
                    f"- Analytics request ID: {self.visualization_request_id}",
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

    def _append_legacy_analysis(self, lines: list[str]) -> None:
        """Preserve the original singleton analysis evidence sections."""

        lines.extend(["", "TREND ANALYSIS:"])

        if self.trend is not None:
            trend = self.trend
            lines.extend(
                [
                    f"- Period column: {trend.period_column}",
                    f"- Value column: {trend.value_column}",
                    f"- First value: {trend.first_value}",
                    f"- Last value: {trend.last_value}",
                    f"- Absolute change: {trend.absolute_change}",
                    f"- Percentage change: {trend.percentage_change}",
                    f"- Direction: {trend.direction}",
                    "- Trend points:",
                ]
            )
            for point in trend.points:
                lines.append(f"  - {point.period}: {point.value}")
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
                        f"    Contribution percentage: {item.contribution_percentage}",
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
                    "- Interpretation constraint: These are deterministic contribution drivers, not proof of causal relationships.",
                ]
            )
            for driver in root_cause.drivers:
                lines.extend(
                    [
                        f"  - Category: {driver.category}",
                        f"    Value: {driver.value}",
                        f"    Contribution percentage: {driver.contribution_percentage}",
                        f"    Direction: {driver.direction}",
                    ]
                )
        else:
            lines.append("- None")

    @staticmethod
    def _format_analysis_result(result: object) -> list[str]:
        """Format a typed analytics result without relying on object repr."""

        if isinstance(result, TrendResult):
            lines = [
                "  Result type: TrendResult",
                f"  Period column: {result.period_column}",
                f"  Value column: {result.value_column}",
                f"  First value: {result.first_value}",
                f"  Last value: {result.last_value}",
                f"  Absolute change: {result.absolute_change}",
                f"  Percentage change: {result.percentage_change}",
                f"  Direction: {result.direction}",
                "  Trend points:",
            ]
            lines.extend(
                f"    - Period: {point.period}; Value: {point.value}"
                for point in result.points
            )
            return lines

        if isinstance(result, PeriodComparison):
            return [
                "  Result type: PeriodComparison",
                f"  Reference value: {result.value_a}",
                f"  Current value: {result.value_b}",
                f"  Absolute change: {result.absolute_change}",
                f"  Percentage change: {result.percentage_change}",
                f"  Direction: {result.direction}",
            ]

        if isinstance(result, VarianceResult):
            return [
                "  Result type: VarianceResult",
                f"  Actual value: {result.actual_value}",
                f"  Reference value: {result.reference_value}",
                f"  Absolute variance: {result.absolute_variance}",
                f"  Percentage variance: {result.percentage_variance}",
                f"  Direction: {result.direction}",
            ]

        if isinstance(result, ContributionResult):
            lines = [
                "  Result type: ContributionResult",
                f"  Dimension column: {result.dimension_column}",
                f"  Value column: {result.value_column}",
                f"  Total change: {result.total_change}",
                "  Contributions:",
            ]
            for item in result.contributions:
                lines.extend(
                    [
                        f"    - Category: {item.category}",
                        f"      Value: {item.value}",
                        f"      Contribution percentage: {item.contribution_percentage}",
                    ]
                )
            return lines

        if isinstance(result, SegmentationResult):
            lines = [
                "  Result type: SegmentationResult",
                f"  Dimension column: {result.dimension_column}",
                f"  Value column: {result.value_column}",
                f"  Total value: {result.total_value}",
                "  Segments:",
            ]
            for segment in result.segments:
                lines.extend(
                    [
                        f"    - Segment: {segment.segment}",
                        f"      Row count: {segment.row_count}",
                        f"      Total value: {segment.total_value}",
                        f"      Average value: {segment.average_value}",
                        f"      Minimum value: {segment.minimum_value}",
                        f"      Maximum value: {segment.maximum_value}",
                    ]
                )
            return lines

        if isinstance(result, RootCauseResult):
            lines = [
                "  Result type: RootCauseResult",
                f"  Dimension column: {result.dimension_column}",
                f"  Value column: {result.value_column}",
                f"  Overall change: {result.overall_change}",
                f"  Overall direction: {result.overall_direction}",
                "  Drivers (contribution decomposition, not proven causes):",
            ]
            for driver in result.drivers:
                lines.extend(
                    [
                        f"    - Category: {driver.category}",
                        f"      Value: {driver.value}",
                        f"      Contribution percentage: {driver.contribution_percentage}",
                        f"      Direction: {driver.direction}",
                    ]
                )
            return lines

        raise TypeError(
            f"Unsupported analytics result type: {type(result).__name__}"
        )


class EvidenceBuilder:
    """Build grounded evidence from deterministic analysis."""

    def __init__(self, currency: str = "INR") -> None:
        if not currency.strip():
            raise ValueError("Currency cannot be empty.")
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
        visualization_source: str | None = None,
        visualization_request_id: str | None = None,
        analytics_execution_result: AnalyticsExecutionResult | None = None,
    ) -> Evidence:
        """Build evidence from deterministic analytical results."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dataframe must be a pandas DataFrame.")

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

        if analytics_execution_result is not None:
            if not isinstance(
                analytics_execution_result,
                AnalyticsExecutionResult,
            ):
                raise TypeError(
                    "Analytics execution result must be an AnalyticsExecutionResult."
                )

            if analytics_executions:
                raise ValueError(
                    "Provide analytics_execution_result or analytics_executions, not both."
                )

            analytics_executions = analytics_execution_result.executions
            skipped_analyses = analytics_execution_result.skipped_analyses
            fatal_errors = analytics_execution_result.fatal_errors
        else:
            skipped_analyses = ()
            fatal_errors = ()

        if visualization_source not in (None, "analytics", "query_result"):
            raise ValueError(
                "Visualization source must be analytics, query_result, or None."
            )

        if visualization_source == "analytics":
            if (
                not isinstance(visualization_request_id, str)
                or not visualization_request_id.strip()
            ):
                raise ValueError(
                    "Analytics visualization requires a request ID."
                )

            if not any(
                execution.request_id == visualization_request_id
                for execution in analytics_executions
            ):
                raise ValueError(
                    "Visualization request ID must match an analytics execution."
                )

        elif visualization_request_id is not None:
            raise ValueError(
                "A visualization request ID requires analytics as its source."
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
            result_columns=tuple(str(column) for column in dataframe.columns),
            currency=self.currency,
            trend=trend,
            period_comparison=period_comparison,
            variance=variance,
            contribution=contribution,
            segmentation=segmentation,
            root_cause=root_cause,
            visualization=visualization,
            analytics_executions=analytics_executions,
            visualization_source=visualization_source,
            visualization_request_id=visualization_request_id,
            skipped_analyses=skipped_analyses,
            fatal_errors=fatal_errors,
        )
