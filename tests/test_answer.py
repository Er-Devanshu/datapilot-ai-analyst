import pandas as pd
import pytest

from datapilot.analytics.answer import AnswerComposer
from datapilot.analytics.evidence import EvidenceBuilder
from datapilot.llm.local import LLMResponse


class FakeLLM:
    """Deterministic fake LLM for AnswerComposer tests."""

    def __init__(
        self,
        response: str = "Total revenue is 600.",
    ) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(
        self,
        prompt: str,
    ) -> LLMResponse:
        self.last_prompt = prompt

        return LLMResponse(
            text=self.response,
            model_name="fake-model",
        )


def build_test_evidence():
    """Build deterministic evidence for AnswerComposer tests."""

    dataframe = pd.DataFrame(
        {
            "region": ["West", "East", "West"],
            "revenue": [100.0, 200.0, 300.0],
        }
    )

    return EvidenceBuilder().build(
        dataframe
    )


def test_answer_composer_returns_grounded_answer() -> None:
    llm = FakeLLM(
        response="Total revenue is ₹600."
    )

    composer = AnswerComposer(llm)

    result = composer.compose(
        question="What is the total revenue?",
        evidence=build_test_evidence(),
    )

    assert result.question == (
        "What is the total revenue?"
    )

    assert result.answer == (
        "Total revenue is ₹600."
    )

    assert result.model_name == "fake-model"


def test_answer_composer_prompt_contains_question_and_evidence() -> None:
    llm = FakeLLM()

    composer = AnswerComposer(llm)

    composer.compose(
        question="What is the total revenue?",
        evidence=build_test_evidence(),
    )

    assert llm.last_prompt is not None

    assert "What is the total revenue?" in (
        llm.last_prompt
    )

    assert "Currency: INR" in (
        llm.last_prompt
    )

    assert "Column: revenue" in (
        llm.last_prompt
    )

    assert "Total: 600.0" in (
        llm.last_prompt
    )

    assert "Average: 200.0" in (
        llm.last_prompt
    )

    assert "Minimum: 100.0" in (
        llm.last_prompt
    )

    assert "Maximum: 300.0" in (
        llm.last_prompt
    )

    assert "Dimension: region" in (
        llm.last_prompt
    )

    assert "Top category: West" in (
        llm.last_prompt
    )

    assert "Bottom category: East" in (
        llm.last_prompt
    )


def test_answer_composer_prompt_requires_currency_from_evidence() -> None:
    llm = FakeLLM()

    composer = AnswerComposer(llm)

    evidence = EvidenceBuilder(
        currency="USD"
    ).build(
        pd.DataFrame(
            {
                "revenue": [100.0, 200.0],
            }
        )
    )

    composer.compose(
        question="What is the revenue?",
        evidence=evidence,
    )

    assert llm.last_prompt is not None
    assert "Currency: USD" in (
        llm.last_prompt
    )


def test_answer_composer_rejects_empty_question() -> None:
    llm = FakeLLM()

    composer = AnswerComposer(llm)

    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        composer.compose(
            question="",
            evidence=build_test_evidence(),
        )


def test_answer_composer_rejects_empty_llm_response() -> None:
    llm = FakeLLM(
        response=""
    )

    composer = AnswerComposer(llm)

    with pytest.raises(
        RuntimeError,
        match="empty answer",
    ):
        composer.compose(
            question="What is the total revenue?",
            evidence=build_test_evidence(),
        )


def test_answer_composer_strips_question_whitespace() -> None:
    llm = FakeLLM()

    composer = AnswerComposer(llm)

    result = composer.compose(
        question="  What is the total revenue?  ",
        evidence=build_test_evidence(),
    )

    assert result.question == (
        "What is the total revenue?"
    )

    assert llm.last_prompt is not None

    assert (
        "USER QUESTION:\n"
        "What is the total revenue?"
        in llm.last_prompt
    )