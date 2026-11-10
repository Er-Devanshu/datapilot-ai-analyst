from dataclasses import FrozenInstanceError

import pytest

from datapilot.analytics.planning import (
    AggregationType,
    AnalysisInputMapping,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlan,
    AnalyticsPlanValidation,
    PlanValidationIssue,
    SkippedAnalysis,
)


def make_request(
    request_id: str = "trend_1",
    analysis_type: AnalysisType = AnalysisType.TREND,
    inputs: AnalysisInputMapping | None = None,
    depends_on: tuple[str, ...] = (),
) -> AnalysisRequest:
    """Build a valid request for contract tests."""

    return AnalysisRequest(
        request_id=request_id,
        analysis_type=analysis_type,
        inputs=inputs or AnalysisInputMapping(),
        depends_on=depends_on,
    )


def test_analysis_type_contains_supported_operations() -> None:
    assert {item.value for item in AnalysisType} == {
        "trend",
        "period_comparison",
        "variance",
        "contribution",
        "segmentation",
        "root_cause",
    }


def test_analysis_input_mapping_accepts_valid_values() -> None:
    mapping = AnalysisInputMapping(
        period_column="month",
        value_column="revenue",
        dimension_column="region",
        reference_period="2025-06",
        current_period="2025-07",
        actual_period="2025-07",
    )

    assert mapping.period_column == "month"
    assert mapping.value_column == "revenue"
    assert mapping.dimension_column == "region"
    assert mapping.reference_period == "2025-06"
    assert mapping.current_period == "2025-07"
    assert mapping.actual_period == "2025-07"


@pytest.mark.parametrize(
    "field_name",
    [
        "period_column",
        "value_column",
        "dimension_column",
        "reference_period",
        "current_period",
        "actual_period",
    ],
)
@pytest.mark.parametrize("invalid_value", ["", "   ", 123])
def test_analysis_input_mapping_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be a non-empty string or None",
    ):
        AnalysisInputMapping(**{field_name: invalid_value})


def test_analysis_input_mapping_accepts_none_values() -> None:
    mapping = AnalysisInputMapping()

    assert mapping.period_column is None
    assert mapping.value_column is None
    assert mapping.dimension_column is None
    assert mapping.reference_period is None
    assert mapping.current_period is None
    assert mapping.actual_period is None


def test_analysis_input_mapping_is_immutable() -> None:
    mapping = AnalysisInputMapping(value_column="revenue")

    with pytest.raises(FrozenInstanceError):
        mapping.value_column = "profit"  # type: ignore[misc]


@pytest.mark.parametrize("invalid_request_id", ["", "   ", None, 123])
def test_analysis_request_rejects_invalid_request_id(
    invalid_request_id: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Analysis request ID must be a non-empty string",
    ):
        AnalysisRequest(
            request_id=invalid_request_id,
            analysis_type=AnalysisType.TREND,
            inputs=AnalysisInputMapping(),
        )


def test_analysis_request_rejects_invalid_analysis_type() -> None:
    with pytest.raises(
        ValueError,
        match="Analysis type must be an AnalysisType value",
    ):
        AnalysisRequest(
            request_id="trend_1",
            analysis_type="trend",  # type: ignore[arg-type]
            inputs=AnalysisInputMapping(),
        )


def test_analysis_request_rejects_invalid_input_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="Analysis inputs must use AnalysisInputMapping",
    ):
        AnalysisRequest(
            request_id="trend_1",
            analysis_type=AnalysisType.TREND,
            inputs={},  # type: ignore[arg-type]
        )


def test_analysis_request_rejects_non_tuple_dependencies() -> None:
    with pytest.raises(
        ValueError,
        match="Analysis dependencies must be provided as a tuple",
    ):
        AnalysisRequest(
            request_id="root_cause_1",
            analysis_type=AnalysisType.ROOT_CAUSE,
            inputs=AnalysisInputMapping(),
            depends_on=["variance_1", "contribution_1"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "dependency_id",
    ["", "   ", None, 123],
)
def test_analysis_request_rejects_invalid_dependency_ids(
    dependency_id: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Every dependency ID must be a non-empty string",
    ):
        AnalysisRequest(
            request_id="root_cause_1",
            analysis_type=AnalysisType.ROOT_CAUSE,
            inputs=AnalysisInputMapping(),
            depends_on=(dependency_id,),  # type: ignore[arg-type]
        )


def test_analysis_request_preserves_dependencies() -> None:
    request = make_request(
        request_id="root_cause_1",
        analysis_type=AnalysisType.ROOT_CAUSE,
        depends_on=("variance_1", "contribution_1"),
    )

    assert request.depends_on == (
        "variance_1",
        "contribution_1",
    )


def test_analysis_request_is_immutable() -> None:
    request = make_request()

    with pytest.raises(FrozenInstanceError):
        request.request_id = "changed"  # type: ignore[misc]


def test_analytics_plan_accepts_valid_requests() -> None:
    trend = make_request()
    contribution = make_request(
        request_id="contribution_1",
        analysis_type=AnalysisType.CONTRIBUTION,
    )

    plan = AnalyticsPlan(
        question="Why did revenue change?",
        requests=(trend, contribution),
    )

    assert plan.question == "Why did revenue change?"
    assert plan.requests == (trend, contribution)


@pytest.mark.parametrize(
    "invalid_question",
    ["", "   ", None, 123],
)
def test_analytics_plan_rejects_invalid_question(
    invalid_question: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Analytics plan question must be a non-empty string",
    ):
        AnalyticsPlan(
            question=invalid_question,
            requests=(),  # type: ignore[arg-type]
        )


def test_analytics_plan_rejects_non_tuple_requests() -> None:
    with pytest.raises(
        ValueError,
        match="Analytics plan requests must be provided as a tuple",
    ):
        AnalyticsPlan(
            question="Analyze revenue",
            requests=[],  # type: ignore[arg-type]
        )


def test_analytics_plan_rejects_non_request_entries() -> None:
    with pytest.raises(
        ValueError,
        match="Every plan entry must be an AnalysisRequest",
    ):
        AnalyticsPlan(
            question="Analyze revenue",
            requests=("not a request",),  # type: ignore[arg-type]
        )


def test_analytics_plan_allows_duplicate_ids_for_validator_handling() -> None:
    first = make_request(request_id="trend_1")
    second = make_request(request_id="trend_1")

    plan = AnalyticsPlan(
        question="Analyze revenue",
        requests=(first, second),
    )

    assert len(plan.requests) == 2
    assert plan.requests[0].request_id == plan.requests[1].request_id


def test_analytics_plan_is_immutable() -> None:
    plan = AnalyticsPlan(
        question="Analyze revenue",
        requests=(make_request(),),
    )

    with pytest.raises(FrozenInstanceError):
        plan.question = "Changed question"  # type: ignore[misc]


def test_plan_validation_issue_accepts_valid_values() -> None:
    issue = PlanValidationIssue(
        code="DEPENDENCY_CYCLE",
        message="The plan contains a dependency cycle.",
        request_id="root_cause_1",
    )

    assert issue.code == "DEPENDENCY_CYCLE"
    assert issue.message == "The plan contains a dependency cycle."
    assert issue.request_id == "root_cause_1"


@pytest.mark.parametrize(
    ("code", "message", "request_id"),
    [
        ("", "A message", None),
        ("   ", "A message", None),
        ("INVALID", "", None),
        ("INVALID", "   ", None),
        ("INVALID", "A message", ""),
        ("INVALID", "A message", "   "),
        ("INVALID", "A message", 123),
    ],
)
def test_plan_validation_issue_rejects_invalid_values(
    code: object,
    message: object,
    request_id: object,
) -> None:
    with pytest.raises(ValueError):
        PlanValidationIssue(
            code=code,
            message=message,
            request_id=request_id,
        )


def test_plan_validation_issue_is_immutable() -> None:
    issue = PlanValidationIssue(
        code="DEPENDENCY_CYCLE",
        message="Dependency cycle detected.",
    )

    with pytest.raises(FrozenInstanceError):
        issue.code = "CHANGED"  # type: ignore[misc]


def test_skipped_analysis_accepts_valid_values() -> None:
    skipped = SkippedAnalysis(
        request_id="trend_1",
        analysis_type=AnalysisType.TREND,
        reason="The requested period column is unavailable.",
    )

    assert skipped.request_id == "trend_1"
    assert skipped.analysis_type == AnalysisType.TREND
    assert skipped.reason == (
        "The requested period column is unavailable."
    )


def test_skipped_analysis_rejects_invalid_request_id() -> None:
    with pytest.raises(
        ValueError,
        match="Skipped analysis request ID must be a non-empty string",
    ):
        SkippedAnalysis(
            request_id=" ",
            analysis_type=AnalysisType.TREND,
            reason="Unsupported request.",
        )


def test_skipped_analysis_rejects_invalid_analysis_type() -> None:
    with pytest.raises(
        ValueError,
        match="Skipped analysis type must be an AnalysisType value",
    ):
        SkippedAnalysis(
            request_id="trend_1",
            analysis_type="trend",  # type: ignore[arg-type]
            reason="Unsupported request.",
        )


def test_skipped_analysis_rejects_invalid_reason() -> None:
    with pytest.raises(
        ValueError,
        match="Skipped analysis reason must be a non-empty string",
    ):
        SkippedAnalysis(
            request_id="trend_1",
            analysis_type=AnalysisType.TREND,
            reason=" ",
        )


def test_skipped_analysis_is_immutable() -> None:
    skipped = SkippedAnalysis(
        request_id="trend_1",
        analysis_type=AnalysisType.TREND,
        reason="Unsupported request.",
    )

    with pytest.raises(FrozenInstanceError):
        skipped.reason = "Changed reason"  # type: ignore[misc]


def test_analytics_plan_validation_accepts_partial_results() -> None:
    accepted = make_request()
    skipped = SkippedAnalysis(
        request_id="unsupported_1",
        analysis_type=AnalysisType.SEGMENTATION,
        reason="Required dimension column is unavailable.",
    )

    validation = AnalyticsPlanValidation(
        accepted_requests=(accepted,),
        skipped_analyses=(skipped,),
    )

    assert validation.accepted_requests == (accepted,)
    assert validation.skipped_analyses == (skipped,)
    assert validation.fatal_errors == ()
    assert validation.is_valid is True


def test_analytics_plan_validation_is_valid_without_fatal_errors() -> None:
    validation = AnalyticsPlanValidation(
        accepted_requests=(make_request(),),
        skipped_analyses=(),
    )

    assert validation.is_valid is True


def test_analytics_plan_validation_is_invalid_with_fatal_errors() -> None:
    issue = PlanValidationIssue(
        code="DUPLICATE_REQUEST_ID",
        message="Request IDs must be unique.",
    )

    validation = AnalyticsPlanValidation(
        accepted_requests=(),
        skipped_analyses=(),
        fatal_errors=(issue,),
    )

    assert validation.fatal_errors == (issue,)
    assert validation.is_valid is False


def test_analytics_plan_validation_rejects_non_tuple_accepted_requests() -> None:
    with pytest.raises(
        ValueError,
        match="Accepted requests must be provided as a tuple",
    ):
        AnalyticsPlanValidation(
            accepted_requests=[],  # type: ignore[arg-type]
            skipped_analyses=(),
        )


def test_analytics_plan_validation_rejects_invalid_accepted_entries() -> None:
    with pytest.raises(
        ValueError,
        match="Accepted requests must contain AnalysisRequest values",
    ):
        AnalyticsPlanValidation(
            accepted_requests=("not a request",),  # type: ignore[arg-type]
            skipped_analyses=(),
        )


def test_analytics_plan_validation_rejects_non_tuple_skipped_analyses() -> None:
    with pytest.raises(
        ValueError,
        match="Skipped analyses must be provided as a tuple",
    ):
        AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=[],  # type: ignore[arg-type]
        )


def test_analytics_plan_validation_rejects_invalid_skipped_entries() -> None:
    with pytest.raises(
        ValueError,
        match="Skipped analyses must contain SkippedAnalysis values",
    ):
        AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=("not skipped",),  # type: ignore[arg-type]
        )


def test_analytics_plan_validation_rejects_non_tuple_fatal_errors() -> None:
    with pytest.raises(
        ValueError,
        match="Fatal errors must be provided as a tuple",
    ):
        AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=(),
            fatal_errors=[],  # type: ignore[arg-type]
        )


def test_analytics_plan_validation_rejects_invalid_fatal_entries() -> None:
    with pytest.raises(
        ValueError,
        match="Fatal errors must contain PlanValidationIssue values",
    ):
        AnalyticsPlanValidation(
            accepted_requests=(),
            skipped_analyses=(),
            fatal_errors=("not an issue",),  # type: ignore[arg-type]
        )


def test_analytics_plan_validation_is_immutable() -> None:
    validation = AnalyticsPlanValidation(
        accepted_requests=(make_request(),),
        skipped_analyses=(),
    )

    with pytest.raises(FrozenInstanceError):
        validation.accepted_requests = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "aggregation",
    [
        AggregationType.SUM,
        AggregationType.MEAN,
        AggregationType.MIN,
        AggregationType.MAX,
        AggregationType.COUNT,
    ],
)
def test_aggregation_type_contains_supported_operations(
    aggregation: AggregationType,
) -> None:
    assert aggregation.value in {"sum", "mean", "min", "max", "count"}


def test_analysis_input_mapping_accepts_aggregation() -> None:
    mapping = AnalysisInputMapping(
        period_column="month",
        value_column="revenue",
        aggregation=AggregationType.SUM,
    )

    assert mapping.aggregation is AggregationType.SUM


def test_analysis_input_mapping_defaults_aggregation_to_none() -> None:
    mapping = AnalysisInputMapping()

    assert mapping.aggregation is None


def test_analysis_input_mapping_rejects_invalid_aggregation() -> None:
    with pytest.raises(
        ValueError,
        match="Aggregation must be an AggregationType value or None",
    ):
        AnalysisInputMapping(aggregation="sum")  # type: ignore[arg-type]


def test_analysis_input_mapping_is_immutable_for_aggregation() -> None:
    mapping = AnalysisInputMapping(aggregation=AggregationType.SUM)

    with pytest.raises(FrozenInstanceError):
        mapping.aggregation = AggregationType.MEAN  # type: ignore[misc]