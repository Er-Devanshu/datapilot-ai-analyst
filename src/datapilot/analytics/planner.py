from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from datapilot.analytics.plan_validator import AnalyticsPlanValidator
from datapilot.analytics.planning import (
    AggregationType,
    AnalysisInputMapping,
    AnalysisRequest,
    AnalysisType,
    AnalyticsPlan,
    AnalyticsPlanValidation,
)
from datapilot.llm.local import LocalLLM
from datapilot.llm.prompts import AnalyticsPlanPromptBuilder


class AnalyticsPlanGenerationError(ValueError):
    """Raised when an LLM response cannot form a valid analytics plan."""


@dataclass(frozen=True)
class AnalyticsPlanningResult:
    """Plan proposal and its deterministic validation outcome."""

    plan: AnalyticsPlan
    validation: AnalyticsPlanValidation
    prompt: str
    model_name: str


class AnalyticsPlanner:
    """Generate and validate structured analytical plans."""

    MAX_TEMPORAL_VALUES = 24

    TEMPORAL_NAME_HINTS = (
        "date",
        "month",
        "period",
        "year",
        "quarter",
        "week",
    )

    def __init__(
        self,
        llm: LocalLLM,
        prompt_builder: AnalyticsPlanPromptBuilder | None = None,
        validator: AnalyticsPlanValidator | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else AnalyticsPlanPromptBuilder()
        )
        self.validator = (
            validator
            if validator is not None
            else AnalyticsPlanValidator()
        )

    def plan(
        self,
        question: str,
        dataframe: pd.DataFrame,
        change_value_columns: set[str] | frozenset[str] | None = None,
    ) -> AnalyticsPlanningResult:
        """Generate an analytics plan and validate it against the result."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Dataframe must be a pandas DataFrame.")

        metadata = self._build_dataframe_metadata(dataframe)

        prompt = self.prompt_builder.build(
            question=question,
            dataframe_metadata=metadata,
            change_value_columns=change_value_columns,
        )

        response = self.llm.generate(prompt)

        if isinstance(response, str):
            response_text = response
            model_name = "unknown"
        else:
            response_text = str(
                getattr(response, "text", "")
            )
            model_name = str(
                getattr(response, "model_name", "unknown")
            )

        plan = self._parse_plan(
            response_text=response_text,
            question=question.strip(),
        )

        validation = self.validator.validate(
            plan=plan,
            dataframe=dataframe,
            change_value_columns=change_value_columns,
        )

        return AnalyticsPlanningResult(
            plan=plan,
            validation=validation,
            prompt=prompt,
            model_name=model_name,
        )

    @classmethod
    def _build_dataframe_metadata(
        cls,
        dataframe: pd.DataFrame,
    ) -> dict[str, object]:
        """Build schema metadata without sending full query-result rows."""

        columns: list[dict[str, object]] = []

        for column in dataframe.columns:
            series = dataframe[column]
            column_name = str(column)

            column_metadata: dict[str, object] = {
                "name": column_name,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
            }

            normalized_name = column_name.lower()

            is_temporal_candidate = any(
                hint in normalized_name
                for hint in cls.TEMPORAL_NAME_HINTS
            )

            if is_temporal_candidate:
                values = (
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )

                column_metadata["observed_values"] = values[
                    : cls.MAX_TEMPORAL_VALUES
                ]
                column_metadata["values_truncated"] = (
                    len(values) > cls.MAX_TEMPORAL_VALUES
                )

            columns.append(column_metadata)

        return {
            "row_count": int(len(dataframe)),
            "columns": columns,
        }

    @classmethod
    def _parse_plan(
        cls,
        response_text: str,
        question: str,
    ) -> AnalyticsPlan:
        """Parse strict JSON into the immutable analytics plan contract."""

        if not isinstance(response_text, str) or not response_text.strip():
            raise AnalyticsPlanGenerationError(
                "The LLM returned an empty analytics plan."
            )

        json_text = cls._remove_code_fence(response_text)

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise AnalyticsPlanGenerationError(
                "The LLM response is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise AnalyticsPlanGenerationError(
                "The analytics plan response must be a JSON object."
            )

        unexpected_keys = set(payload) - {"requests"}

        if unexpected_keys:
            raise AnalyticsPlanGenerationError(
                "Unexpected analytics plan fields: "
                + ", ".join(sorted(str(key) for key in unexpected_keys))
            )

        requests_payload = payload.get("requests")

        if not isinstance(requests_payload, list):
            raise AnalyticsPlanGenerationError(
                "The analytics plan must contain a requests array."
            )

        requests = tuple(
            cls._parse_request(item, index)
            for index, item in enumerate(requests_payload)
        )

        try:
            return AnalyticsPlan(
                question=question,
                requests=requests,
            )
        except (TypeError, ValueError) as exc:
            raise AnalyticsPlanGenerationError(
                f"The analytics plan contract is invalid: {exc}"
            ) from exc

    @classmethod
    def _parse_request(
        cls,
        payload: Any,
        index: int,
    ) -> AnalysisRequest:
        """Parse and validate one proposed analysis request."""

        if not isinstance(payload, dict):
            raise AnalyticsPlanGenerationError(
                f"Request at index {index} must be a JSON object."
            )

        allowed_request_keys = {
            "request_id",
            "analysis_type",
            "inputs",
            "depends_on",
        }

        unexpected_keys = set(payload) - allowed_request_keys

        if unexpected_keys:
            raise AnalyticsPlanGenerationError(
                f"Request at index {index} contains unsupported fields: "
                + ", ".join(sorted(str(key) for key in unexpected_keys))
            )

        request_id = payload.get("request_id")
        analysis_type_value = payload.get("analysis_type")
        inputs_payload = payload.get("inputs", {})
        dependencies_payload = payload.get("depends_on", [])

        if not isinstance(request_id, str) or not request_id.strip():
            raise AnalyticsPlanGenerationError(
                f"Request at index {index} requires a non-empty request_id."
            )

        if not isinstance(analysis_type_value, str):
            raise AnalyticsPlanGenerationError(
                f"Request '{request_id}' requires an analysis_type string."
            )

        if not isinstance(inputs_payload, dict):
            raise AnalyticsPlanGenerationError(
                f"Inputs for request '{request_id}' must be a JSON object."
            )

        if not isinstance(dependencies_payload, list):
            raise AnalyticsPlanGenerationError(
                f"Dependencies for request '{request_id}' must be a JSON array."
            )

        if not all(
            isinstance(dependency, str) and dependency.strip()
            for dependency in dependencies_payload
        ):
            raise AnalyticsPlanGenerationError(
                f"Dependencies for request '{request_id}' "
                "must contain non-empty strings."
            )

        allowed_input_keys = {
            "period_column",
            "value_column",
            "dimension_column",
            "reference_period",
            "current_period",
            "actual_period",
            "aggregation",
        }

        unexpected_input_keys = set(inputs_payload) - allowed_input_keys

        if unexpected_input_keys:
            raise AnalyticsPlanGenerationError(
                f"Inputs for request '{request_id}' contain unsupported "
                "fields: "
                + ", ".join(
                    sorted(str(key) for key in unexpected_input_keys)
                )
            )

        try:
            analysis_type = AnalysisType(analysis_type_value)
        except ValueError as exc:
            raise AnalyticsPlanGenerationError(
                f"Request '{request_id}' uses unsupported analysis type: "
                f"{analysis_type_value}"
            ) from exc

        inputs = dict(inputs_payload)
        aggregation_value = inputs.get("aggregation")

        if aggregation_value is not None:
            if not isinstance(aggregation_value, str):
                raise AnalyticsPlanGenerationError(
                    f"Aggregation for request '{request_id}' "
                    "must be a string or null."
                )

            try:
                inputs["aggregation"] = AggregationType(
                    aggregation_value
                )
            except ValueError as exc:
                raise AnalyticsPlanGenerationError(
                    f"Request '{request_id}' uses unsupported aggregation: "
                    f"{aggregation_value}"
                ) from exc

        try:
            mapping = AnalysisInputMapping(**inputs)

            return AnalysisRequest(
                request_id=request_id,
                analysis_type=analysis_type,
                inputs=mapping,
                depends_on=tuple(dependencies_payload),
            )
        except (TypeError, ValueError) as exc:
            raise AnalyticsPlanGenerationError(
                f"Request '{request_id}' is invalid: {exc}"
            ) from exc

    @staticmethod
    def _remove_code_fence(response_text: str) -> str:
        """Remove one optional JSON Markdown code fence."""

        text = response_text.strip()

        if not text.startswith("```"):
            return text

        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            raise AnalyticsPlanGenerationError(
                "The LLM returned an incomplete JSON code fence."
            )

        return match.group(1).strip()