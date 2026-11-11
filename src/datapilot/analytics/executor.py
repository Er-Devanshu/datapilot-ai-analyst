
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.contribution import (
    ContributionAnalyzer,
    ContributionResult,
)
from datapilot.analytics.period import (
    PeriodComparator,
    PeriodComparison,
)
from datapilot.analytics.planning import (
    AggregationType,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlanValidation,
    SkippedAnalysis,
)
from datapilot.analytics.root_cause import (
    RootCauseAnalyzer,
    RootCauseResult,
)
from datapilot.analytics.segmentation import (
    SegmentationAnalyzer,
    SegmentationResult,
)
from datapilot.analytics.trend import (
    TrendAnalyzer,
    TrendResult,
)
from datapilot.analytics.variance import (
    VarianceAnalyzer,
    VarianceResult,
)


AnalysisOutput = (
    TrendResult
    | PeriodComparison
    | VarianceResult
    | ContributionResult
    | SegmentationResult
    | RootCauseResult
)


@dataclass(frozen=True)
class AnalysisExecution:
    """One successfully executed analytical request."""

    request_id: str
    analysis_type: AnalysisType
    result: AnalysisOutput


@dataclass(frozen=True)
class AnalyticsExecutionResult:
    """Successful and skipped outcomes from analytics execution."""

    executions: tuple[AnalysisExecution, ...]
    skipped_analyses: tuple[SkippedAnalysis, ...]
    fatal_errors: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.executions, tuple):
            raise ValueError("Executions must be provided as a tuple.")

        if not isinstance(self.skipped_analyses, tuple):
            raise ValueError(
                "Skipped analyses must be provided as a tuple."
            )

        if not isinstance(self.fatal_errors, tuple):
            raise ValueError("Fatal errors must be provided as a tuple.")

    def get_result(self, request_id: str) -> AnalysisOutput | None:
        """Return the successful result for a request ID."""

        for execution in self.executions:
            if execution.request_id == request_id:
                return execution.result

        return None


class AnalyticsExecutor:
    """Execute validated analytics requests deterministically."""

    def __init__(
        self,
        trend_analyzer: TrendAnalyzer | None = None,
        period_comparator: PeriodComparator | None = None,
        variance_analyzer: VarianceAnalyzer | None = None,
        contribution_analyzer: ContributionAnalyzer | None = None,
        segmentation_analyzer: SegmentationAnalyzer | None = None,
        root_cause_analyzer: RootCauseAnalyzer | None = None,
    ) -> None:
        self.trend_analyzer = trend_analyzer or TrendAnalyzer()
        self.period_comparator = (
            period_comparator or PeriodComparator()
        )
        self.variance_analyzer = (
            variance_analyzer or VarianceAnalyzer()
        )
        self.contribution_analyzer = (
            contribution_analyzer or ContributionAnalyzer()
        )
        self.segmentation_analyzer = (
            segmentation_analyzer or SegmentationAnalyzer()
        )
        self.root_cause_analyzer = (
            root_cause_analyzer or RootCauseAnalyzer()
        )

    def execute(
        self,
        dataframe: pd.DataFrame,
        validation: AnalyticsPlanValidation,
    ) -> AnalyticsExecutionResult:
        """Execute only requests accepted by deterministic validation."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dataframe must be a pandas DataFrame.")

        if not isinstance(validation, AnalyticsPlanValidation):
            raise TypeError(
                "Validation must be an AnalyticsPlanValidation."
            )

        if validation.fatal_errors:
            return AnalyticsExecutionResult(
                executions=(),
                skipped_analyses=validation.skipped_analyses,
                fatal_errors=validation.fatal_errors,
            )

        successful: dict[str, AnalysisExecution] = {}
        skipped = list(validation.skipped_analyses)

        for request in validation.accepted_requests:
            failed_dependencies = [
                dependency_id
                for dependency_id in request.depends_on
                if dependency_id not in successful
            ]

            if failed_dependencies:
                skipped.append(
                    SkippedAnalysis(
                        request_id=request.request_id,
                        analysis_type=request.analysis_type,
                        reason=(
                            "A required dependency did not execute "
                            "successfully: "
                            + ", ".join(failed_dependencies)
                        ),
                    )
                )
                continue

            try:
                result = self._execute_request(
                    dataframe=dataframe,
                    request=request,
                    successful=successful,
                )
            except Exception as exc:
                skipped.append(
                    SkippedAnalysis(
                        request_id=request.request_id,
                        analysis_type=request.analysis_type,
                        reason=(
                            "Analytical execution failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                )
                continue

            successful[request.request_id] = AnalysisExecution(
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                result=result,
            )

        executions = tuple(
            successful[request.request_id]
            for request in validation.accepted_requests
            if request.request_id in successful
        )

        return AnalyticsExecutionResult(
            executions=executions,
            skipped_analyses=tuple(skipped),
            fatal_errors=(),
        )

    def _execute_request(
        self,
        dataframe: pd.DataFrame,
        request: AnalysisRequest,
        successful: dict[str, AnalysisExecution],
    ) -> AnalysisOutput:
        """Dispatch one accepted request to its deterministic analyzer."""

        inputs = request.inputs

        if request.analysis_type == AnalysisType.TREND:
            return self.trend_analyzer.analyze(
                dataframe=dataframe,
                period_column=inputs.period_column,
                value_column=inputs.value_column,
            )

        if request.analysis_type == AnalysisType.PERIOD_COMPARISON:
            reference_value = self._aggregate_period(
                dataframe=dataframe,
                period_column=inputs.period_column,
                period_label=inputs.reference_period,
                value_column=inputs.value_column,
                aggregation=inputs.aggregation,
            )
            current_value = self._aggregate_period(
                dataframe=dataframe,
                period_column=inputs.period_column,
                period_label=inputs.current_period,
                value_column=inputs.value_column,
                aggregation=inputs.aggregation,
            )

            return self.period_comparator.compare(
                value_a=reference_value,
                value_b=current_value,
            )

        if request.analysis_type == AnalysisType.VARIANCE:
            reference_value = self._aggregate_period(
                dataframe=dataframe,
                period_column=inputs.period_column,
                period_label=inputs.reference_period,
                value_column=inputs.value_column,
                aggregation=inputs.aggregation,
            )
            actual_value = self._aggregate_period(
                dataframe=dataframe,
                period_column=inputs.period_column,
                period_label=inputs.actual_period,
                value_column=inputs.value_column,
                aggregation=inputs.aggregation,
            )

            return self.variance_analyzer.analyze(
                actual_value=actual_value,
                reference_value=reference_value,
            )

        if request.analysis_type == AnalysisType.CONTRIBUTION:
            return self.contribution_analyzer.analyze_change(
                dataframe=dataframe,
                dimension_column=inputs.dimension_column,
                value_column=inputs.value_column,
            )

        if request.analysis_type == AnalysisType.SEGMENTATION:
            return self.segmentation_analyzer.analyze(
                dataframe=dataframe,
                dimension_column=inputs.dimension_column,
                value_column=inputs.value_column,
            )

        if request.analysis_type == AnalysisType.ROOT_CAUSE:
            dependencies = [
                successful[dependency_id].result
                for dependency_id in request.depends_on
            ]

            variance_results = [
                item
                for item in dependencies
                if isinstance(item, VarianceResult)
            ]
            contribution_results = [
                item
                for item in dependencies
                if isinstance(item, ContributionResult)
            ]

            if (
                len(variance_results) != 1
                or len(contribution_results) != 1
            ):
                raise ValueError(
                    "Root-cause dependencies did not produce "
                    "one variance and one contribution result."
                )

            return self.root_cause_analyzer.analyze(
                variance=variance_results[0],
                contribution=contribution_results[0],
            )

        raise ValueError(
            f"Unsupported analysis type: {request.analysis_type}"
        )

    @staticmethod
    def _aggregate_period(
        dataframe: pd.DataFrame,
        period_column: str,
        period_label: str,
        value_column: str,
        aggregation: AggregationType,
    ) -> float:
        """Aggregate one period using the explicitly selected operation."""

        period_mask = (
            dataframe[period_column].astype(str) == period_label
        )
        values = dataframe.loc[period_mask, value_column].dropna()

        if aggregation == AggregationType.COUNT:
            return float(values.count())

        numeric_values = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna()

        if numeric_values.empty:
            raise ValueError(
                f"Period '{period_label}' has no valid numeric "
                f"observations in column '{value_column}'."
            )

        if aggregation == AggregationType.SUM:
            value = numeric_values.sum()
        elif aggregation == AggregationType.MEAN:
            value = numeric_values.mean()
        elif aggregation == AggregationType.MIN:
            value = numeric_values.min()
        elif aggregation == AggregationType.MAX:
            value = numeric_values.max()
        else:
            raise ValueError(
                f"Unsupported aggregation: {aggregation}"
            )

        return float(value)
