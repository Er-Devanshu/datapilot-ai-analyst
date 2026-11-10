
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datapilot.analytics.planning import (
    AggregationType,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlan,
    AnalyticsPlanValidation,
    PlanValidationIssue,
    SkippedAnalysis,
)


@dataclass(frozen=True)
class _RequestValidation:
    """Internal validation outcome for one analysis request."""

    is_valid: bool
    reason: str | None = None


class AnalyticsPlanValidator:
    """Validate proposed analytics plans deterministically.

    The validator checks:
    - Plan-level dependency integrity.
    - Operation-specific input mappings.
    - Query-result column availability and data types.
    - Explicit aggregation requirements for period comparisons and variance.
    - Trusted semantic metadata for contribution analysis.
    - Root-cause dependency requirements.
    - Deterministic execution ordering.

    It does not execute analytical operations or aggregate data.
    """

    def validate(
        self,
        plan: AnalyticsPlan,
        dataframe: pd.DataFrame,
        change_value_columns: set[str] | frozenset[str] | None = None,
    ) -> AnalyticsPlanValidation:
        """Validate a plan against query results and trusted metadata."""

        if not isinstance(plan, AnalyticsPlan):
            raise TypeError("Plan must be an AnalyticsPlan.")

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dataframe must be a pandas DataFrame.")

        if change_value_columns is None:
            trusted_change_columns: set[str] = set()
        else:
            if not isinstance(change_value_columns, (set, frozenset)):
                raise ValueError(
                    "Change value columns must be a set or frozenset."
                )

            if not all(
                isinstance(column, str) and column.strip()
                for column in change_value_columns
            ):
                raise ValueError(
                    "Trusted change column names must be non-empty strings."
                )

            trusted_change_columns = set(change_value_columns)

        fatal_errors = self._validate_structure(plan)

        if fatal_errors:
            return AnalyticsPlanValidation(
                accepted_requests=(),
                skipped_analyses=(),
                fatal_errors=tuple(fatal_errors),
            )

        if dataframe.empty:
            return AnalyticsPlanValidation(
                accepted_requests=(),
                skipped_analyses=(),
                fatal_errors=(
                    PlanValidationIssue(
                        code="EMPTY_DATAFRAME",
                        message=(
                            "The query result contains no rows. "
                            "Analytical requests cannot be validated."
                        ),
                    ),
                ),
            )

        requests_by_id = {
            request.request_id: request
            for request in plan.requests
        }

        skipped_by_id: dict[str, SkippedAnalysis] = {}

        for request in plan.requests:
            outcome = self._validate_request(
                request=request,
                dataframe=dataframe,
                requests_by_id=requests_by_id,
                change_value_columns=trusted_change_columns,
            )

            if not outcome.is_valid:
                skipped_by_id[request.request_id] = SkippedAnalysis(
                    request_id=request.request_id,
                    analysis_type=request.analysis_type,
                    reason=outcome.reason or "Request validation failed.",
                )

        self._propagate_skipped_dependencies(
            plan=plan,
            skipped_by_id=skipped_by_id,
        )

        accepted_requests = self._execution_order(
            plan=plan,
            skipped_request_ids=set(skipped_by_id),
        )

        skipped_analyses = tuple(
            skipped_by_id[request.request_id]
            for request in plan.requests
            if request.request_id in skipped_by_id
        )

        return AnalyticsPlanValidation(
            accepted_requests=accepted_requests,
            skipped_analyses=skipped_analyses,
        )

    def _validate_structure(
        self,
        plan: AnalyticsPlan,
    ) -> list[PlanValidationIssue]:
        """Detect plan-level dependency errors."""

        issues: list[PlanValidationIssue] = []

        request_ids = [
            request.request_id
            for request in plan.requests
        ]

        duplicate_ids = {
            request_id
            for request_id in request_ids
            if request_ids.count(request_id) > 1
        }

        for request_id in sorted(duplicate_ids):
            issues.append(
                PlanValidationIssue(
                    code="DUPLICATE_REQUEST_ID",
                    message=(
                        f"Analysis request ID is duplicated: {request_id}"
                    ),
                    request_id=request_id,
                )
            )

        if duplicate_ids:
            return issues

        known_ids = set(request_ids)

        for request in plan.requests:
            duplicate_dependencies = (
                len(request.depends_on) != len(set(request.depends_on))
            )

            if duplicate_dependencies:
                issues.append(
                    PlanValidationIssue(
                        code="DUPLICATE_DEPENDENCY",
                        message=(
                            "Analysis dependencies contain duplicate IDs."
                        ),
                        request_id=request.request_id,
                    )
                )

            for dependency_id in request.depends_on:
                if dependency_id not in known_ids:
                    issues.append(
                        PlanValidationIssue(
                            code="MISSING_DEPENDENCY",
                            message=(
                                f"Dependency does not exist: {dependency_id}"
                            ),
                            request_id=request.request_id,
                        )
                    )

        if issues:
            return issues

        cycle = self._find_dependency_cycle(plan)

        if cycle:
            issues.append(
                PlanValidationIssue(
                    code="DEPENDENCY_CYCLE",
                    message=(
                        "The analytics plan contains a dependency cycle: "
                        + " -> ".join(cycle)
                    ),
                )
            )

        return issues

    def _find_dependency_cycle(
        self,
        plan: AnalyticsPlan,
    ) -> tuple[str, ...] | None:
        """Find a dependency cycle using depth-first traversal."""

        requests_by_id = {
            request.request_id: request
            for request in plan.requests
        }

        visiting: set[str] = set()
        visited: set[str] = set()
        path: list[str] = []

        def visit(request_id: str) -> tuple[str, ...] | None:
            if request_id in visiting:
                cycle_start = path.index(request_id)
                return tuple(path[cycle_start:] + [request_id])

            if request_id in visited:
                return None

            visiting.add(request_id)
            path.append(request_id)

            request = requests_by_id[request_id]

            for dependency_id in request.depends_on:
                cycle = visit(dependency_id)

                if cycle:
                    return cycle

            path.pop()
            visiting.remove(request_id)
            visited.add(request_id)

            return None

        for request in plan.requests:
            cycle = visit(request.request_id)

            if cycle:
                return cycle

        return None

    def _validate_request(
        self,
        request: AnalysisRequest,
        dataframe: pd.DataFrame,
        requests_by_id: dict[str, AnalysisRequest],
        change_value_columns: set[str],
    ) -> _RequestValidation:
        """Validate the operation-specific requirements."""

        inputs = request.inputs

        if request.analysis_type == AnalysisType.TREND:
            if request.depends_on:
                return self._invalid(
                    "Trend analysis cannot declare dependencies."
                )

            if inputs.aggregation is not None:
                return self._invalid(
                    "Trend analysis does not accept aggregation metadata."
                )

            error = self._require_columns(
                dataframe,
                inputs.period_column,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            error = self._require_numeric_column(
                dataframe,
                inputs.value_column,
            )

            return self._invalid(error) if error else self._valid()

        if request.analysis_type == AnalysisType.PERIOD_COMPARISON:
            if request.depends_on:
                return self._invalid(
                    "Period comparison cannot declare dependencies."
                )

            error = self._require_aggregation(inputs.aggregation)

            if error:
                return self._invalid(error)

            error = self._require_columns(
                dataframe,
                inputs.period_column,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            if inputs.aggregation != AggregationType.COUNT:
                error = self._require_numeric_column(
                    dataframe,
                    inputs.value_column,
                )

                if error:
                    return self._invalid(error)

            error = self._require_period_labels(
                dataframe,
                inputs.period_column,
                inputs.reference_period,
                inputs.current_period,
            )

            return self._invalid(error) if error else self._valid()

        if request.analysis_type == AnalysisType.VARIANCE:
            if request.depends_on:
                return self._invalid(
                    "Variance analysis cannot declare dependencies."
                )

            error = self._require_aggregation(inputs.aggregation)

            if error:
                return self._invalid(error)

            error = self._require_columns(
                dataframe,
                inputs.period_column,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            if inputs.aggregation != AggregationType.COUNT:
                error = self._require_numeric_column(
                    dataframe,
                    inputs.value_column,
                )

                if error:
                    return self._invalid(error)

            error = self._require_period_labels(
                dataframe,
                inputs.period_column,
                inputs.reference_period,
                inputs.actual_period,
            )

            return self._invalid(error) if error else self._valid()

        if request.analysis_type == AnalysisType.CONTRIBUTION:
            if request.depends_on:
                return self._invalid(
                    "Contribution analysis cannot declare dependencies."
                )

            if inputs.aggregation is not None:
                return self._invalid(
                    "Contribution analysis does not accept aggregation metadata."
                )

            error = self._require_columns(
                dataframe,
                inputs.dimension_column,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            error = self._require_numeric_column(
                dataframe,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            if inputs.value_column not in change_value_columns:
                return self._invalid(
                    "Contribution analysis requires a value column "
                    "explicitly trusted as category-level changes."
                )

            return self._valid()

        if request.analysis_type == AnalysisType.SEGMENTATION:
            if request.depends_on:
                return self._invalid(
                    "Segmentation cannot declare dependencies."
                )

            if inputs.aggregation is not None:
                return self._invalid(
                    "Segmentation does not accept aggregation metadata."
                )

            error = self._require_columns(
                dataframe,
                inputs.dimension_column,
                inputs.value_column,
            )

            if error:
                return self._invalid(error)

            error = self._require_numeric_column(
                dataframe,
                inputs.value_column,
            )

            return self._invalid(error) if error else self._valid()

        if request.analysis_type == AnalysisType.ROOT_CAUSE:
            return self._validate_root_cause_dependencies(
                request=request,
                requests_by_id=requests_by_id,
            )

        return self._invalid(
            f"Unsupported analysis type: {request.analysis_type}"
        )

    def _require_aggregation(
        self,
        aggregation: AggregationType | None,
    ) -> str | None:
        """Require an explicitly selected supported aggregation."""

        if aggregation is None:
            return (
                "An explicit aggregation is required for period comparison "
                "and variance analysis."
            )

        if not isinstance(aggregation, AggregationType):
            return "The requested aggregation type is unsupported."

        return None

    def _validate_root_cause_dependencies(
        self,
        request: AnalysisRequest,
        requests_by_id: dict[str, AnalysisRequest],
    ) -> _RequestValidation:
        """Require one variance and one contribution dependency."""

        if any(
            value is not None
            for value in (
                request.inputs.period_column,
                request.inputs.value_column,
                request.inputs.dimension_column,
                request.inputs.reference_period,
                request.inputs.current_period,
                request.inputs.actual_period,
                request.inputs.aggregation,
            )
        ):
            return self._invalid(
                "Root-cause analysis does not accept direct column "
                "or period mappings. It consumes dependency results."
            )

        if len(request.depends_on) != 2:
            return self._invalid(
                "Root-cause analysis requires exactly two dependencies: "
                "one variance and one contribution analysis."
            )

        dependency_types = [
            requests_by_id[dependency_id].analysis_type
            for dependency_id in request.depends_on
        ]

        if dependency_types.count(AnalysisType.VARIANCE) != 1:
            return self._invalid(
                "Root-cause analysis requires exactly one variance dependency."
            )

        if dependency_types.count(AnalysisType.CONTRIBUTION) != 1:
            return self._invalid(
                "Root-cause analysis requires exactly one contribution "
                "dependency."
            )

        return self._valid()

    def _require_columns(
        self,
        dataframe: pd.DataFrame,
        *column_names: str | None,
    ) -> str | None:
        """Check that required column names exist."""

        for column_name in column_names:
            if column_name is None:
                return "A required input column mapping is missing."

            if column_name not in dataframe.columns:
                return f"Required column does not exist: {column_name}"

        return None

    def _require_numeric_column(
        self,
        dataframe: pd.DataFrame,
        column_name: str | None,
    ) -> str | None:
        """Check that a mapped measure is numeric."""

        if column_name is None:
            return "A required numeric value column mapping is missing."

        if not pd.api.types.is_numeric_dtype(dataframe[column_name]):
            return f"Value column must be numeric: {column_name}"

        return None

    def _require_period_labels(
        self,
        dataframe: pd.DataFrame,
        period_column: str | None,
        *period_labels: str | None,
    ) -> str | None:
        """Check that requested periods occur in the result."""

        if period_column is None:
            return "A period column mapping is required."

        available_periods = {
            str(value)
            for value in dataframe[period_column].dropna().unique()
        }

        for period_label in period_labels:
            if period_label is None:
                return "A required period label is missing."

            if period_label not in available_periods:
                return (
                    "Requested period does not exist in the result: "
                    f"{period_label}"
                )

        if len(set(period_labels)) != len(period_labels):
            return "Comparison periods must be distinct."

        return None

    def _propagate_skipped_dependencies(
        self,
        plan: AnalyticsPlan,
        skipped_by_id: dict[str, SkippedAnalysis],
    ) -> None:
        """Skip requests whose dependencies cannot be executed."""

        changed = True

        while changed:
            changed = False

            for request in plan.requests:
                if request.request_id in skipped_by_id:
                    continue

                failed_dependencies = [
                    dependency_id
                    for dependency_id in request.depends_on
                    if dependency_id in skipped_by_id
                ]

                if not failed_dependencies:
                    continue

                skipped_by_id[request.request_id] = SkippedAnalysis(
                    request_id=request.request_id,
                    analysis_type=request.analysis_type,
                    reason=(
                        "A required dependency was skipped: "
                        + ", ".join(failed_dependencies)
                    ),
                )

                changed = True

    def _execution_order(
        self,
        plan: AnalyticsPlan,
        skipped_request_ids: set[str],
    ) -> tuple[AnalysisRequest, ...]:
        """Return accepted requests in stable topological order."""

        accepted_by_id = {
            request.request_id: request
            for request in plan.requests
            if request.request_id not in skipped_request_ids
        }

        original_order = {
            request.request_id: index
            for index, request in enumerate(plan.requests)
        }

        remaining_dependencies = {
            request_id: {
                dependency_id
                for dependency_id in request.depends_on
                if dependency_id in accepted_by_id
            }
            for request_id, request in accepted_by_id.items()
        }

        ordered: list[AnalysisRequest] = []

        while remaining_dependencies:
            ready_ids = sorted(
                (
                    request_id
                    for request_id, dependencies
                    in remaining_dependencies.items()
                    if not dependencies
                ),
                key=original_order.__getitem__,
            )

            if not ready_ids:
                break

            for request_id in ready_ids:
                ordered.append(accepted_by_id[request_id])
                del remaining_dependencies[request_id]

            for dependencies in remaining_dependencies.values():
                dependencies.difference_update(ready_ids)

        return tuple(ordered)

    @staticmethod
    def _valid() -> _RequestValidation:
        return _RequestValidation(is_valid=True)

    @staticmethod
    def _invalid(reason: str) -> _RequestValidation:
        return _RequestValidation(
            is_valid=False,
            reason=reason,
        )
