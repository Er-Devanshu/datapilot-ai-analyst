import json

import pandas as pd
import pytest

from datapilot.analytics.planner import (
    AnalyticsPlanGenerationError,
    AnalyticsPlanner,
)
from datapilot.analytics.planning import (
    AggregationType,
    AnalysisType,
)
from datapilot.llm.prompts import AnalyticsPlanPromptBuilder


class FakeLLM:
    """Deterministic LLM test double."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


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


def trend_response() -> str:
    return json.dumps(
        {
            "requests": [
                {
                    "request_id": "trend_1",
                    "analysis_type": "trend",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                    },
                    "depends_on": [],
                }
            ]
        }
    )


def comparison_response() -> str:
    return json.dumps(
        {
            "requests": [
                {
                    "request_id": "comparison_1",
                    "analysis_type": "period_comparison",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "reference_period": "2025-06",
                        "current_period": "2025-07",
                        "aggregation": "sum",
                    },
                    "depends_on": [],
                }
            ]
        }
    )


def test_prompt_builder_includes_question_and_metadata() -> None:
    builder = AnalyticsPlanPromptBuilder()

    prompt = builder.build(
        question="Show revenue trend.",
        dataframe_metadata={
            "row_count": 10,
            "columns": [
                {
                    "name": "month",
                    "dtype": "object",
                },
                {
                    "name": "revenue",
                    "dtype": "float64",
                },
            ],
        },
    )

    assert "Show revenue trend." in prompt
    assert '"row_count": 10' in prompt
    assert '"name": "month"' in prompt
    assert "RETURN THE JSON OBJECT NOW" in prompt
    assert "valid JSON only" in prompt


def test_prompt_builder_includes_trusted_change_columns() -> None:
    builder = AnalyticsPlanPromptBuilder()

    prompt = builder.build(
        question="Explain the revenue change by region.",
        dataframe_metadata={"columns": []},
        change_value_columns={"revenue_change"},
    )

    assert "revenue_change" in prompt
    assert "APPLICATION-PROVIDED" in prompt


def test_prompt_builder_rejects_empty_question() -> None:
    builder = AnalyticsPlanPromptBuilder()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        builder.build(
            question=" ",
            dataframe_metadata={},
        )


def test_prompt_builder_rejects_invalid_metadata_type() -> None:
    builder = AnalyticsPlanPromptBuilder()

    with pytest.raises(TypeError, match="dictionary"):
        builder.build(
            question="Analyze revenue.",
            dataframe_metadata=[],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_columns",
    [
        ["revenue_change"],
        {""},
        {123},
    ],
)
def test_prompt_builder_rejects_invalid_trusted_columns(
    invalid_columns: object,
) -> None:
    builder = AnalyticsPlanPromptBuilder()

    with pytest.raises(ValueError):
        builder.build(
            question="Analyze revenue.",
            dataframe_metadata={},
            change_value_columns=invalid_columns,  # type: ignore[arg-type]
        )


def test_planner_builds_temporal_metadata_without_sending_numeric_values(
    dataframe: pd.DataFrame,
) -> None:
    metadata = AnalyticsPlanner._build_dataframe_metadata(dataframe)

    assert metadata["row_count"] == 4

    columns = metadata["columns"]
    assert isinstance(columns, list)

    month_metadata = next(
        item for item in columns if item["name"] == "month"
    )
    revenue_metadata = next(
        item for item in columns if item["name"] == "revenue"
    )

    assert month_metadata["observed_values"] == [
        "2025-06",
        "2025-07",
    ]
    assert "observed_values" not in revenue_metadata


def test_planner_generates_and_validates_trend(
    dataframe: pd.DataFrame,
) -> None:
    llm = FakeLLM(trend_response())
    planner = AnalyticsPlanner(llm)

    result = planner.plan(
        question="Show the revenue trend.",
        dataframe=dataframe,
    )

    assert result.plan.question == "Show the revenue trend."
    assert len(result.plan.requests) == 1
    assert result.plan.requests[0].analysis_type is AnalysisType.TREND

    assert result.validation.is_valid is True
    assert len(result.validation.accepted_requests) == 1
    assert result.validation.skipped_analyses == ()

    assert result.model_name == "unknown"
    assert llm.last_prompt is not None
    assert "Show the revenue trend." in llm.last_prompt


def test_planner_converts_aggregation_to_enum(
    dataframe: pd.DataFrame,
) -> None:
    planner = AnalyticsPlanner(FakeLLM(comparison_response()))

    result = planner.plan(
        question="Compare June and July revenue.",
        dataframe=dataframe,
    )

    request = result.plan.requests[0]

    assert request.analysis_type is AnalysisType.PERIOD_COMPARISON
    assert request.inputs.aggregation is AggregationType.SUM
    assert result.validation.accepted_requests == (request,)


def test_planner_accepts_json_inside_markdown_fence(
    dataframe: pd.DataFrame,
) -> None:
    response = f"```json\n{trend_response()}\n```"
    planner = AnalyticsPlanner(FakeLLM(response))

    result = planner.plan(
        question="Show the revenue trend.",
        dataframe=dataframe,
    )

    assert len(result.validation.accepted_requests) == 1


def test_planner_returns_skipped_request_for_unknown_column(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "trend_1",
                    "analysis_type": "trend",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "unknown_measure",
                    },
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    result = planner.plan(
        question="Show the unknown measure trend.",
        dataframe=dataframe,
    )

    assert result.validation.is_valid is True
    assert result.validation.accepted_requests == ()
    assert len(result.validation.skipped_analyses) == 1
    assert "unknown_measure" in (
        result.validation.skipped_analyses[0].reason
    )


def test_planner_passes_trusted_change_columns_to_validator(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "contribution_1",
                    "analysis_type": "contribution",
                    "inputs": {
                        "dimension_column": "region",
                        "value_column": "revenue_change",
                    },
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    result = planner.plan(
        question="Explain revenue change by region.",
        dataframe=dataframe,
        change_value_columns={"revenue_change"},
    )

    assert len(result.validation.accepted_requests) == 1
    assert result.validation.skipped_analyses == ()


def test_planner_skips_contribution_without_trusted_metadata(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "contribution_1",
                    "analysis_type": "contribution",
                    "inputs": {
                        "dimension_column": "region",
                        "value_column": "revenue_change",
                    },
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    result = planner.plan(
        question="Explain revenue change by region.",
        dataframe=dataframe,
    )

    assert result.validation.accepted_requests == ()
    assert len(result.validation.skipped_analyses) == 1


def test_planner_rejects_invalid_json(
    dataframe: pd.DataFrame,
) -> None:
    planner = AnalyticsPlanner(FakeLLM("{invalid json"))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="not valid JSON",
    ):
        planner.plan(
            question="Show revenue trend.",
            dataframe=dataframe,
        )


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
    ],
)
def test_planner_rejects_empty_response(
    dataframe: pd.DataFrame,
    response: str,
) -> None:
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="empty analytics plan",
    ):
        planner.plan(
            question="Show revenue trend.",
            dataframe=dataframe,
        )


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        '{"unexpected": []}',
        '{"requests": {}}',
    ],
)
def test_planner_rejects_invalid_plan_shapes(
    dataframe: pd.DataFrame,
    response: str,
) -> None:
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(AnalyticsPlanGenerationError):
        planner.plan(
            question="Show revenue trend.",
            dataframe=dataframe,
        )


def test_planner_rejects_unsupported_analysis_type(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "custom_1",
                    "analysis_type": "forecast",
                    "inputs": {},
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="unsupported analysis type",
    ):
        planner.plan(
            question="Forecast revenue.",
            dataframe=dataframe,
        )


def test_planner_rejects_unsupported_aggregation(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "comparison_1",
                    "analysis_type": "period_comparison",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "reference_period": "2025-06",
                        "current_period": "2025-07",
                        "aggregation": "median",
                    },
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="unsupported aggregation",
    ):
        planner.plan(
            question="Compare June and July revenue.",
            dataframe=dataframe,
        )


def test_planner_rejects_unexpected_request_fields(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "trend_1",
                    "analysis_type": "trend",
                    "inputs": {},
                    "depends_on": [],
                    "execute_python": "print('unsafe')",
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="unsupported fields",
    ):
        planner.plan(
            question="Show revenue trend.",
            dataframe=dataframe,
        )


def test_planner_rejects_unexpected_input_fields(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "trend_1",
                    "analysis_type": "trend",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "sql": "SELECT * FROM data",
                    },
                    "depends_on": [],
                }
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    with pytest.raises(
        AnalyticsPlanGenerationError,
        match="unsupported fields",
    ):
        planner.plan(
            question="Show revenue trend.",
            dataframe=dataframe,
        )


def test_planner_returns_fatal_validation_for_duplicate_ids(
    dataframe: pd.DataFrame,
) -> None:
    response = json.dumps(
        {
            "requests": [
                {
                    "request_id": "duplicate",
                    "analysis_type": "trend",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                    },
                    "depends_on": [],
                },
                {
                    "request_id": "duplicate",
                    "analysis_type": "trend",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                    },
                    "depends_on": [],
                },
            ]
        }
    )
    planner = AnalyticsPlanner(FakeLLM(response))

    result = planner.plan(
        question="Show revenue trend.",
        dataframe=dataframe,
    )

    assert result.validation.is_valid is False
    assert result.validation.fatal_errors[0].code == (
        "DUPLICATE_REQUEST_ID"
    )


def test_planner_rejects_empty_question(
    dataframe: pd.DataFrame,
) -> None:
    planner = AnalyticsPlanner(FakeLLM(trend_response()))

    with pytest.raises(ValueError, match="Question cannot be empty"):
        planner.plan(
            question=" ",
            dataframe=dataframe,
        )


def test_planner_rejects_invalid_dataframe() -> None:
    planner = AnalyticsPlanner(FakeLLM(trend_response()))

    with pytest.raises(TypeError, match="pandas DataFrame"):
        planner.plan(
            question="Show revenue trend.",
            dataframe=[],  # type: ignore[arg-type]
        )