
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.executor import AnalyticsExecutionResult
from datapilot.visualization.analytics_adapter import (
    AnalyticsVisualizationAdapter,
)
from datapilot.visualization.models import VisualizationSpec
from datapilot.visualization.selector import (
    VisualizationSelector,
)


@dataclass(frozen=True)
class AnalyticsVisualizationSelection:
    """Selected visualization and its source metadata."""

    specification: VisualizationSpec
    dataframe: pd.DataFrame
    source: str
    request_id: str | None
    reason: str


class AnalyticsAwareVisualizationSelector:
    """Prefer non-empty supported analytics outputs over raw query results."""

    def __init__(
        self,
        adapter: AnalyticsVisualizationAdapter | None = None,
        fallback_selector: VisualizationSelector | None = None,
    ) -> None:
        self.adapter = adapter or AnalyticsVisualizationAdapter()
        self.fallback_selector = (
            fallback_selector or VisualizationSelector()
        )

    def select(
        self,
        dataframe: pd.DataFrame,
        analytics_execution: AnalyticsExecutionResult | None = None,
    ) -> AnalyticsVisualizationSelection:
        """Select the first non-empty supported analytics result or use fallback."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Dataframe must be a pandas DataFrame."
            )

        if analytics_execution is not None:
            if not isinstance(
                analytics_execution,
                AnalyticsExecutionResult,
            ):
                raise TypeError(
                    "Analytics execution must be an "
                    "AnalyticsExecutionResult or None."
                )

            for execution in analytics_execution.executions:
                adapted = self.adapter.adapt(execution)

                if adapted is None or adapted.dataframe.empty:
                    continue

                specification = VisualizationSpec(
                    chart_type=adapted.chart_type,
                    x_column=adapted.x_column,
                    y_column=adapted.y_column,
                    color_column=adapted.color_column,
                    title=adapted.title,
                )

                return AnalyticsVisualizationSelection(
                    specification=specification,
                    dataframe=adapted.dataframe,
                    source="analytics",
                    request_id=adapted.request_id,
                    reason=(
                        "Visualization selected from deterministic "
                        f"{adapted.analysis_type} analysis."
                    ),
                )

        fallback = self.fallback_selector.select(dataframe)

        return AnalyticsVisualizationSelection(
            specification=fallback.specification,
            dataframe=dataframe,
            source="query_result",
            request_id=None,
            reason=(
                "No non-empty supported analytics visualization "
                "was available. "
                f"{fallback.reason}"
            ),
        )
