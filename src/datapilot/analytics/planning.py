
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnalysisType(StrEnum):
    """Supported deterministic analytical operations."""

    TREND = "trend"
    PERIOD_COMPARISON = "period_comparison"
    VARIANCE = "variance"
    CONTRIBUTION = "contribution"
    SEGMENTATION = "segmentation"
    ROOT_CAUSE = "root_cause"


class AggregationType(StrEnum):
    """Supported deterministic aggregation operations."""

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


@dataclass(frozen=True)
class AnalysisInputMapping:
    """Explicit source-column, period, and aggregation mappings."""

    period_column: str | None = None
    value_column: str | None = None
    dimension_column: str | None = None
    reference_period: str | None = None
    current_period: str | None = None
    actual_period: str | None = None
    aggregation: AggregationType | None = None

    def __post_init__(self) -> None:
        """Validate optional mapping values."""

        for field_name in (
            "period_column",
            "value_column",
            "dimension_column",
            "reference_period",
            "current_period",
            "actual_period",
        ):
            value = getattr(self, field_name)

            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty string or None."
                )

        if self.aggregation is not None and not isinstance(
            self.aggregation,
            AggregationType,
        ):
            raise ValueError(
                "Aggregation must be an AggregationType value or None."
            )


@dataclass(frozen=True)
class AnalysisRequest:
    """One proposed deterministic analytical operation."""

    request_id: str
    analysis_type: AnalysisType
    inputs: AnalysisInputMapping
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the request's local structure."""

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError(
                "Analysis request ID must be a non-empty string."
            )

        if not isinstance(self.analysis_type, AnalysisType):
            raise ValueError(
                "Analysis type must be an AnalysisType value."
            )

        if not isinstance(self.inputs, AnalysisInputMapping):
            raise ValueError(
                "Analysis inputs must use AnalysisInputMapping."
            )

        if not isinstance(self.depends_on, tuple):
            raise ValueError(
                "Analysis dependencies must be provided as a tuple."
            )

        for dependency_id in self.depends_on:
            if (
                not isinstance(dependency_id, str)
                or not dependency_id.strip()
            ):
                raise ValueError(
                    "Every dependency ID must be a non-empty string."
                )


@dataclass(frozen=True)
class AnalyticsPlan:
    """Structured proposal for deterministic analytical work."""

    question: str
    requests: tuple[AnalysisRequest, ...]

    def __post_init__(self) -> None:
        """Validate the plan's basic structure."""

        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError(
                "Analytics plan question must be a non-empty string."
            )

        if not isinstance(self.requests, tuple):
            raise ValueError(
                "Analytics plan requests must be provided as a tuple."
            )

        if not all(
            isinstance(request, AnalysisRequest)
            for request in self.requests
        ):
            raise ValueError(
                "Every plan entry must be an AnalysisRequest."
            )


@dataclass(frozen=True)
class PlanValidationIssue:
    """A fatal issue that prevents safe interpretation of a plan."""

    code: str
    message: str
    request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate issue details."""

        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError(
                "Validation issue code must be a non-empty string."
            )

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError(
                "Validation issue message must be a non-empty string."
            )

        if self.request_id is not None and (
            not isinstance(self.request_id, str)
            or not self.request_id.strip()
        ):
            raise ValueError(
                "Validation issue request ID must be a non-empty string or None."
            )


@dataclass(frozen=True)
class SkippedAnalysis:
    """An individual analysis request rejected by deterministic validation."""

    request_id: str
    analysis_type: AnalysisType
    reason: str

    def __post_init__(self) -> None:
        """Validate skipped-analysis details."""

        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError(
                "Skipped analysis request ID must be a non-empty string."
            )

        if not isinstance(self.analysis_type, AnalysisType):
            raise ValueError(
                "Skipped analysis type must be an AnalysisType value."
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "Skipped analysis reason must be a non-empty string."
            )


@dataclass(frozen=True)
class AnalyticsPlanValidation:
    """Result of deterministic analytics-plan validation."""

    accepted_requests: tuple[AnalysisRequest, ...]
    skipped_analyses: tuple[SkippedAnalysis, ...]
    fatal_errors: tuple[PlanValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        """Validate the validation result's structure."""

        if not isinstance(self.accepted_requests, tuple):
            raise ValueError(
                "Accepted requests must be provided as a tuple."
            )

        if not all(
            isinstance(request, AnalysisRequest)
            for request in self.accepted_requests
        ):
            raise ValueError(
                "Accepted requests must contain AnalysisRequest values."
            )

        if not isinstance(self.skipped_analyses, tuple):
            raise ValueError(
                "Skipped analyses must be provided as a tuple."
            )

        if not all(
            isinstance(item, SkippedAnalysis)
            for item in self.skipped_analyses
        ):
            raise ValueError(
                "Skipped analyses must contain SkippedAnalysis values."
            )

        if not isinstance(self.fatal_errors, tuple):
            raise ValueError(
                "Fatal errors must be provided as a tuple."
            )

        if not all(
            isinstance(issue, PlanValidationIssue)
            for issue in self.fatal_errors
        ):
            raise ValueError(
                "Fatal errors must contain PlanValidationIssue values."
            )

    @property
    def is_valid(self) -> bool:
        """Whether the plan is structurally safe to interpret."""

        return not self.fatal_errors
