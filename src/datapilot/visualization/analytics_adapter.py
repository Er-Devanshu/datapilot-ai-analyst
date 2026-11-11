
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.contribution import ContributionResult
from datapilot.analytics.executor import AnalysisExecution
from datapilot.analytics.period import PeriodComparison
from datapilot.analytics.root_cause import RootCauseResult
from datapilot.analytics.segmentation import SegmentationResult
from datapilot.analytics.trend import TrendResult
from datapilot.analytics.variance import VarianceResult


@dataclass(frozen=True)
class AnalyticsVisualizationData:
    """Visualization-ready data derived from one analytics execution."""

    request_id: str
    analysis_type: str
    dataframe: pd.DataFrame
    title: str
    chart_type: str
    x_column: str | None
    y_column: str | None
    color_column: str | None = None


class AnalyticsVisualizationAdapter:
    """Convert deterministic analytics results into chart-ready data."""

    def adapt(
        self,
        execution: AnalysisExecution,
    ) -> AnalyticsVisualizationData | None:
        """Convert one supported analytical execution into tabular data."""

        if not isinstance(execution, AnalysisExecution):
            raise TypeError(
                "Execution must be an AnalysisExecution."
            )

        result = execution.result

        if isinstance(result, TrendResult):
            dataframe = pd.DataFrame(
                [
                    {
                        "period": point.period,
                        "value": point.value,
                    }
                    for point in result.points
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title=f"{result.value_column} Trend",
                chart_type="line",
                x_column="period",
                y_column="value",
            )

        if isinstance(result, SegmentationResult):
            dataframe = pd.DataFrame(
                [
                    {
                        "segment": segment.segment,
                        "total_value": segment.total_value,
                    }
                    for segment in result.segments
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title=f"{result.value_column} by {result.dimension_column}",
                chart_type="bar",
                x_column="segment",
                y_column="total_value",
            )

        if isinstance(result, ContributionResult):
            dataframe = pd.DataFrame(
                [
                    {
                        "category": item.category,
                        "value": item.value,
                    }
                    for item in result.contributions
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title=f"Contribution to {result.value_column} Change",
                chart_type="bar",
                x_column="category",
                y_column="value",
            )

        if isinstance(result, RootCauseResult):
            dataframe = pd.DataFrame(
                [
                    {
                        "category": driver.category,
                        "value": driver.value,
                        "direction": driver.direction,
                    }
                    for driver in result.drivers
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title=f"Drivers of {result.value_column} Change",
                chart_type="bar",
                x_column="category",
                y_column="value",
                color_column="direction",
            )

        if isinstance(result, PeriodComparison):
            dataframe = pd.DataFrame(
                [
                    {
                        "metric": "Reference value",
                        "value": result.value_a,
                    },
                    {
                        "metric": "Current value",
                        "value": result.value_b,
                    },
                    {
                        "metric": "Absolute change",
                        "value": result.absolute_change,
                    },
                    {
                        "metric": "Percentage change",
                        "value": result.percentage_change,
                    },
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title="Period Comparison",
                chart_type="table",
                x_column=None,
                y_column=None,
            )

        if isinstance(result, VarianceResult):
            dataframe = pd.DataFrame(
                [
                    {
                        "metric": "Actual value",
                        "value": result.actual_value,
                    },
                    {
                        "metric": "Reference value",
                        "value": result.reference_value,
                    },
                    {
                        "metric": "Absolute variance",
                        "value": result.absolute_variance,
                    },
                    {
                        "metric": "Percentage variance",
                        "value": result.percentage_variance,
                    },
                ]
            )

            return self._build(
                execution=execution,
                dataframe=dataframe,
                title="Variance Analysis",
                chart_type="table",
                x_column=None,
                y_column=None,
            )

        return None

    @staticmethod
    def _build(
        execution: AnalysisExecution,
        dataframe: pd.DataFrame,
        title: str,
        chart_type: str,
        x_column: str | None,
        y_column: str | None,
        color_column: str | None = None,
    ) -> AnalyticsVisualizationData:
        return AnalyticsVisualizationData(
            request_id=execution.request_id,
            analysis_type=execution.analysis_type.value,
            dataframe=dataframe,
            title=title,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
        )
