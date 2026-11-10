import pandas as pd
import pytest

from datapilot.analytics.plan_validator import AnalyticsPlanValidator
from datapilot.analytics.planning import (
    AggregationType,
    AnalysisInputMapping,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlan,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": [
                "2025-06",
                "2025-07",
                "2025-06",
                "2025-07",
            ],
            "revenue": [100.0, 120.0, 80.0, 90.0],
            "revenue_change": [-10.0, 20.0, -5.0, 15.0],
            "region": ["North", "North", "South", "South"],
        }
    )


@pytest.fixture
def validator() -> AnalyticsPlanValidator:
    return AnalyticsPlanValidator()


def make_request(
    request_id: str,
    analysis_type: AnalysisType,
    inputs: AnalysisInputMapping | None = None,
    depends_on: tuple[str, ...] = (),
) -> AnalysisRequest:
    return AnalysisRequest(
        request_id=request_id,
        analysis_type=analysis_type,
        inputs=inputs or AnalysisInputMapping(),
        depends_on=depends_on,
    )


def make_plan(*requests: AnalysisRequest) -> AnalyticsPlan:
    return AnalyticsPlan(
        question="Why did revenue change?",
        requests=tuple(requests),
    )


def trend_request(request_id: str = "trend_1") -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.TREND,
        inputs=AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
        ),
    )


def comparison_request(
    request_id: str = "comparison_1",
) -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.PERIOD_COMPARISON,
        inputs=AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-06",
            current_period="2025-07",
        ),
    )


def variance_request(request_id: str = "variance_1") -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.VARIANCE,
        inputs=AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-06",
            actual_period="2025-07",
        ),
    )


def contribution_request(
    request_id: str = "contribution_1",
) -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.CONTRIBUTION,
        inputs=AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue_change",
        ),
    )


def segmentation_request(
    request_id: str = "segmentation_1",
) -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.SEGMENTATION,
        inputs=AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue",
        ),
    )


def root_cause_request(
    request_id: str = "root_cause_1",
    depends_on: tuple[str, ...] = (
        "variance_1",
        "contribution_1",
    ),
) -> AnalysisRequest:
    return make_request(
        request_id=request_id,
        analysis_type=AnalysisType.ROOT_CAUSE,
        depends_on=depends_on,
    )


def test_validator_accepts_valid_trend(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = trend_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == (request,)
    assert result.skipped_analyses == ()
    assert result.fatal_errors == ()


def test_validator_accepts_valid_period_comparison(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = comparison_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == (request,)
    assert result.skipped_analyses == ()


def test_validator_accepts_valid_variance(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = variance_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == (request,)


def test_validator_accepts_valid_contribution_with_trusted_metadata(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = contribution_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.is_valid is True
    assert result.accepted_requests == (request,)
    assert result.skipped_analyses == ()


def test_validator_rejects_contribution_without_trusted_metadata(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = contribution_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == ()
    assert len(result.skipped_analyses) == 1
    assert result.skipped_analyses[0].request_id == "contribution_1"
    assert "category-level changes" in result.skipped_analyses[0].reason


def test_validator_accepts_valid_segmentation(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = segmentation_request()

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == (request,)


@pytest.mark.parametrize(
    "analysis_request",
    [
        make_request(
            "trend_1",
            AnalysisType.TREND,
            AnalysisInputMapping(value_column="revenue"),
        ),
        make_request(
            "trend_1",
            AnalysisType.TREND,
            AnalysisInputMapping(
                period_column="missing_month",
                value_column="revenue",
            ),
        ),
        make_request(
            "trend_1",
            AnalysisType.TREND,
            AnalysisInputMapping(
                period_column="month",
                value_column="missing_revenue",
            ),
        ),
    ],
)
def test_validator_rejects_invalid_trend_mappings(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    analysis_request: AnalysisRequest,
) -> None:
    result = validator.validate(
        plan=make_plan(analysis_request),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == ()
    assert len(result.skipped_analyses) == 1


def test_validator_rejects_non_numeric_measure(
    validator: AnalyticsPlanValidator,
) -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["2025-06", "2025-07"],
            "revenue": ["100", "120"],
        }
    )

    result = validator.validate(
        plan=make_plan(trend_request()),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()
    assert "must be numeric" in result.skipped_analyses[0].reason


def test_validator_rejects_missing_comparison_period(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "comparison_1",
        AnalysisType.PERIOD_COMPARISON,
        AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-05",
            current_period="2025-07",
        ),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()
    assert "2025-05" in result.skipped_analyses[0].reason


def test_validator_rejects_identical_comparison_periods(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "comparison_1",
        AnalysisType.PERIOD_COMPARISON,
        AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-06",
            current_period="2025-06",
        ),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()
    assert "distinct" in result.skipped_analyses[0].reason


def test_validator_rejects_missing_variance_period(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "variance_1",
        AnalysisType.VARIANCE,
        AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-06",
            actual_period="2025-08",
        ),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()
    assert "2025-08" in result.skipped_analyses[0].reason


def test_validator_rejects_contribution_using_ordinary_measure(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "contribution_1",
        AnalysisType.CONTRIBUTION,
        AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue",
        ),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.accepted_requests == ()
    assert "category-level changes" in result.skipped_analyses[0].reason


def test_validator_rejects_missing_dimension_column(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "segmentation_1",
        AnalysisType.SEGMENTATION,
        AnalysisInputMapping(
            dimension_column="missing_region",
            value_column="revenue",
        ),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()
    assert "missing_region" in result.skipped_analyses[0].reason


def test_validator_rejects_missing_root_cause_dependencies(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = root_cause_request(depends_on=("variance_1",))
    variance = variance_request()

    result = validator.validate(
        plan=make_plan(variance, request),
        dataframe=dataframe,
    )

    assert result.accepted_requests == (variance,)
    assert len(result.skipped_analyses) == 1
    assert result.skipped_analyses[0].request_id == "root_cause_1"
    assert "exactly two dependencies" in result.skipped_analyses[0].reason


@pytest.mark.parametrize(
    "dependency_types",
    [
        (AnalysisType.VARIANCE, AnalysisType.SEGMENTATION),
        (AnalysisType.CONTRIBUTION, AnalysisType.SEGMENTATION),
        (AnalysisType.VARIANCE, AnalysisType.VARIANCE),
    ],
)
def test_validator_rejects_incompatible_root_cause_dependencies(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    dependency_types: tuple[AnalysisType, AnalysisType],
) -> None:
    first = make_request("dependency_1", dependency_types[0])
    second = make_request("dependency_2", dependency_types[1])
    root_cause = root_cause_request(
        depends_on=("dependency_1", "dependency_2"),
    )

    result = validator.validate(
        plan=make_plan(first, second, root_cause),
        dataframe=dataframe,
    )

    assert result.accepted_requests == ()


def test_validator_rejects_direct_root_cause_mappings(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    variance = variance_request()
    contribution = contribution_request()

    root_cause = make_request(
        "root_cause_1",
        AnalysisType.ROOT_CAUSE,
        AnalysisInputMapping(value_column="revenue"),
        ("variance_1", "contribution_1"),
    )

    result = validator.validate(
        plan=make_plan(variance, contribution, root_cause),
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.accepted_requests == (variance, contribution)
    assert len(result.skipped_analyses) == 1
    assert "does not accept direct column" in result.skipped_analyses[0].reason


def test_validator_rejects_duplicate_request_ids(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    first = trend_request("duplicate")
    second = segmentation_request("duplicate")

    result = validator.validate(
        plan=make_plan(first, second),
        dataframe=dataframe,
    )

    assert result.is_valid is False
    assert result.accepted_requests == ()
    assert result.skipped_analyses == ()
    assert result.fatal_errors[0].code == "DUPLICATE_REQUEST_ID"


def test_validator_rejects_missing_dependency(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = root_cause_request(
        depends_on=("missing_variance", "missing_contribution"),
    )

    result = validator.validate(
        plan=make_plan(request),
        dataframe=dataframe,
    )

    assert result.is_valid is False
    assert result.accepted_requests == ()
    assert {
        issue.code for issue in result.fatal_errors
    } == {"MISSING_DEPENDENCY"}


def test_validator_rejects_duplicate_dependencies(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    request = make_request(
        "root_cause_1",
        AnalysisType.ROOT_CAUSE,
        depends_on=("variance_1", "variance_1"),
    )
    variance = variance_request()

    result = validator.validate(
        plan=make_plan(variance, request),
        dataframe=dataframe,
    )

    assert result.is_valid is False
    assert result.fatal_errors[0].code == "DUPLICATE_DEPENDENCY"


def test_validator_detects_dependency_cycle(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    first = make_request(
        "first",
        AnalysisType.TREND,
        AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
        ),
        depends_on=("second",),
    )

    second = make_request(
        "second",
        AnalysisType.SEGMENTATION,
        AnalysisInputMapping(
            dimension_column="region",
            value_column="revenue",
        ),
        depends_on=("first",),
    )

    result = validator.validate(
        plan=make_plan(first, second),
        dataframe=dataframe,
    )

    assert result.is_valid is False
    assert result.fatal_errors[0].code == "DEPENDENCY_CYCLE"


def test_validator_preserves_independent_valid_requests(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    valid_trend = trend_request()
    invalid_segmentation = make_request(
        "segmentation_1",
        AnalysisType.SEGMENTATION,
        AnalysisInputMapping(
            dimension_column="missing_region",
            value_column="revenue",
        ),
    )
    valid_variance = variance_request()

    result = validator.validate(
        plan=make_plan(
            valid_trend,
            invalid_segmentation,
            valid_variance,
        ),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == (valid_trend, valid_variance)
    assert len(result.skipped_analyses) == 1
    assert result.skipped_analyses[0].request_id == "segmentation_1"


def test_validator_propagates_skipped_dependencies(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    invalid_variance = make_request(
        "variance_1",
        AnalysisType.VARIANCE,
        AnalysisInputMapping(
            period_column="month",
            value_column="missing_revenue",
            aggregation=AggregationType.SUM,
            reference_period="2025-06",
            actual_period="2025-07",
        ),
    )
    contribution = contribution_request()
    root_cause = root_cause_request()
    independent_trend = trend_request()

    result = validator.validate(
        plan=make_plan(
            invalid_variance,
            contribution,
            root_cause,
            independent_trend,
        ),
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.is_valid is True
    assert result.accepted_requests == (contribution, independent_trend)

    skipped_ids = {
        item.request_id for item in result.skipped_analyses
    }
    assert skipped_ids == {"variance_1", "root_cause_1"}


def test_validator_orders_dependencies_before_root_cause(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    root_cause = root_cause_request()
    contribution = contribution_request()
    variance = variance_request()

    result = validator.validate(
        plan=make_plan(root_cause, contribution, variance),
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.is_valid is True
    assert [
        request.request_id for request in result.accepted_requests
    ] == [
        "contribution_1",
        "variance_1",
        "root_cause_1",
    ]


def test_validator_accepts_empty_plan(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    result = validator.validate(
        plan=make_plan(),
        dataframe=dataframe,
    )

    assert result.is_valid is True
    assert result.accepted_requests == ()
    assert result.skipped_analyses == ()


def test_validator_rejects_empty_dataframe(
    validator: AnalyticsPlanValidator,
) -> None:
    result = validator.validate(
        plan=make_plan(trend_request()),
        dataframe=pd.DataFrame(columns=["month", "revenue"]),
    )

    assert result.is_valid is False
    assert result.fatal_errors[0].code == "EMPTY_DATAFRAME"


def test_validator_rejects_invalid_plan_type(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="AnalyticsPlan"):
        validator.validate(
            plan="invalid",  # type: ignore[arg-type]
            dataframe=dataframe,
        )


def test_validator_rejects_invalid_dataframe_type(
    validator: AnalyticsPlanValidator,
) -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validator.validate(
            plan=make_plan(trend_request()),
            dataframe=[],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_columns",
    [
        ["revenue_change"],
        {"", "revenue_change"},
        {123},
    ],
)
def test_validator_rejects_invalid_trusted_metadata(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    invalid_columns: object,
) -> None:
    with pytest.raises(ValueError):
        validator.validate(
            plan=make_plan(contribution_request()),
            dataframe=dataframe,
            change_value_columns=invalid_columns,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "analysis_type, inputs",
    [
        (
            AnalysisType.PERIOD_COMPARISON,
            AnalysisInputMapping(
                period_column="month",
                value_column="revenue",
                reference_period="2025-06",
                current_period="2025-07",
            ),
        ),
        (
            AnalysisType.VARIANCE,
            AnalysisInputMapping(
                period_column="month",
                value_column="revenue",
                reference_period="2025-06",
                actual_period="2025-07",
            ),
        ),
    ],
)
def test_validator_skips_period_operations_without_explicit_aggregation(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    analysis_type: AnalysisType,
    inputs: AnalysisInputMapping,
) -> None:
    request = make_request("missing_aggregation", analysis_type, inputs)
    result = validator.validate(make_plan(request), dataframe)

    assert result.is_valid is True
    assert result.accepted_requests == ()
    assert len(result.skipped_analyses) == 1
    assert "explicit aggregation is required" in result.skipped_analyses[0].reason


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
def test_validator_accepts_supported_period_aggregations(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    aggregation: AggregationType,
) -> None:
    request = make_request(
        "comparison_aggregation",
        AnalysisType.PERIOD_COMPARISON,
        AnalysisInputMapping(
            period_column="month",
            value_column="revenue",
            reference_period="2025-06",
            current_period="2025-07",
            aggregation=aggregation,
        ),
    )
    result = validator.validate(make_plan(request), dataframe)

    assert result.accepted_requests == (request,)
    assert result.skipped_analyses == ()


def test_validator_allows_count_aggregation_on_non_numeric_measure(
    validator: AnalyticsPlanValidator,
) -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["2025-06", "2025-07"],
            "invoice_id": ["A", "B"],
        }
    )
    request = make_request(
        "count_comparison",
        AnalysisType.PERIOD_COMPARISON,
        AnalysisInputMapping(
            period_column="month",
            value_column="invoice_id",
            reference_period="2025-06",
            current_period="2025-07",
            aggregation=AggregationType.COUNT,
        ),
    )
    result = validator.validate(make_plan(request), dataframe)

    assert result.accepted_requests == (request,)


def test_validator_rejects_numeric_aggregation_on_non_numeric_measure(
    validator: AnalyticsPlanValidator,
) -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["2025-06", "2025-07"],
            "invoice_id": ["A", "B"],
        }
    )
    request = make_request(
        "sum_comparison",
        AnalysisType.PERIOD_COMPARISON,
        AnalysisInputMapping(
            period_column="month",
            value_column="invoice_id",
            reference_period="2025-06",
            current_period="2025-07",
            aggregation=AggregationType.SUM,
        ),
    )
    result = validator.validate(make_plan(request), dataframe)

    assert result.accepted_requests == ()
    assert "must be numeric" in result.skipped_analyses[0].reason


@pytest.mark.parametrize(
    "analysis_type, inputs",
    [
        (
            AnalysisType.TREND,
            AnalysisInputMapping(
                period_column="month",
                value_column="revenue",
                aggregation=AggregationType.SUM,
            ),
        ),
        (
            AnalysisType.CONTRIBUTION,
            AnalysisInputMapping(
                dimension_column="region",
                value_column="revenue_change",
                aggregation=AggregationType.SUM,
            ),
        ),
        (
            AnalysisType.SEGMENTATION,
            AnalysisInputMapping(
                dimension_column="region",
                value_column="revenue",
                aggregation=AggregationType.SUM,
            ),
        ),
    ],
)
def test_validator_rejects_aggregation_for_other_analysis_types(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
    analysis_type: AnalysisType,
    inputs: AnalysisInputMapping,
) -> None:
    request = make_request("unexpected_aggregation", analysis_type, inputs)
    result = validator.validate(
        make_plan(request),
        dataframe,
        change_value_columns={"revenue_change"},
    )

    assert result.accepted_requests == ()
    assert (
        "does not accept aggregation metadata"
        in result.skipped_analyses[0].reason
    )


def test_validator_rejects_aggregation_on_root_cause_request(
    validator: AnalyticsPlanValidator,
    dataframe: pd.DataFrame,
) -> None:
    variance = variance_request()
    contribution = contribution_request()
    root_cause = make_request(
        "root_cause_1",
        AnalysisType.ROOT_CAUSE,
        AnalysisInputMapping(aggregation=AggregationType.SUM),
        ("variance_1", "contribution_1"),
    )
    result = validator.validate(
        make_plan(variance, contribution, root_cause),
        dataframe,
        change_value_columns={"revenue_change"},
    )

    assert [
        item.request_id for item in result.accepted_requests
    ] == [
        "variance_1",
        "contribution_1",
    ]
    assert result.skipped_analyses[-1].request_id == "root_cause_1"