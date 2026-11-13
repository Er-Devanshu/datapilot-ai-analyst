import pandas as pd

from datapilot.analytics.answer import AnswerComposer
from datapilot.analytics.answer_validation import (
    AnswerValidationResult,
)
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.llm.local import LLMResponse


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.response,
            model_name="fake-model",
        )


def build_evidence():
    return EvidenceBuilder(
        currency="INR",
    ).build(
        pd.DataFrame(
            {
                "revenue": [100.0, 200.0, 300.0],
            }
        )
    )


def test_composer_returns_validation_result_for_supported_answer():
    composer = AnswerComposer(
        FakeLLM("Total revenue is ₹600.")
    )

    result = composer.compose(
        question="What is the total revenue?",
        evidence=build_evidence(),
    )

    assert result.answer == "Total revenue is ₹600."
    assert result.validation is not None
    assert isinstance(result.validation, AnswerValidationResult)
    assert result.validation.valid
    assert result.validation.issues == ()


def test_composer_reports_unsupported_number_without_changing_answer():
    original_answer = "Total revenue is ₹600 and profit is ₹900."

    composer = AnswerComposer(
        FakeLLM(original_answer)
    )

    result = composer.compose(
        question="What is the total revenue?",
        evidence=build_evidence(),
    )

    assert result.answer == original_answer
    assert result.validation is not None
    assert not result.validation.valid
    assert any(
        issue.code == "unsupported_number"
        and issue.offending_text == "900"
        for issue in result.validation.issues
    )


def test_composer_reports_explicit_currency_mismatch_without_changing_answer():
    original_answer = "Total revenue is US$600."

    composer = AnswerComposer(
        FakeLLM(original_answer)
    )

    result = composer.compose(
        question="What is the total revenue?",
        evidence=build_evidence(),
    )

    assert result.answer == original_answer
    assert result.validation is not None
    assert any(
        issue.code == "currency_mismatch"
        for issue in result.validation.issues
    )


def test_composer_does_not_infer_currency_from_standalone_dollar():
    original_answer = "Total revenue is $600."

    composer = AnswerComposer(
        FakeLLM(original_answer)
    )

    result = composer.compose(
        question="What is the total revenue?",
        evidence=build_evidence(),
    )

    assert result.answer == original_answer
    assert result.validation is not None
    assert not any(
        issue.code == "currency_mismatch"
        for issue in result.validation.issues
    )


def test_composer_preserves_model_name_and_question():
    composer = AnswerComposer(
        FakeLLM("Total revenue is ₹600.")
    )

    result = composer.compose(
        question="  What is the total revenue?  ",
        evidence=build_evidence(),
    )

    assert result.question == "What is the total revenue?"
    assert result.model_name == "fake-model"
    assert result.validation is not None