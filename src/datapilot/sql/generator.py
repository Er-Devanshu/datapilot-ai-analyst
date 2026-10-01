from __future__ import annotations

from dataclasses import dataclass

from datapilot.llm.local import LocalLLM
from datapilot.llm.prompts import SQLPromptBuilder


@dataclass(frozen=True)
class SQLGenerationResult:
    """Result produced by SQL generation."""

    sql: str
    prompt: str


class SQLGenerator:
    """Generate SQL from a business question and schema context."""

    def __init__(
        self,
        llm: LocalLLM,
        prompt_builder: SQLPromptBuilder | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = (
            prompt_builder
            if prompt_builder is not None
            else SQLPromptBuilder()
        )

    def generate(
        self,
        question: str,
        schema_context: str,
    ) -> SQLGenerationResult:
        """Generate a SQL query for a business question."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        prompt = self.prompt_builder.build(
            question=question,
            schema_context=schema_context,
        )

        response = self.llm.generate(prompt)

        # The production LocalLLM returns a response object with
        # a .text attribute, while deterministic test doubles may
        # return a plain string. Support both contracts.
        if isinstance(response, str):
            response_text = response
        else:
            response_text = response.text

        sql = self._clean_sql(
            response_text
        )

        if not sql:
            raise RuntimeError(
                "The LLM returned an empty SQL query."
            )

        return SQLGenerationResult(
            sql=sql,
            prompt=prompt,
        )

    @staticmethod
    def _clean_sql(
        text: str,
    ) -> str:
        """Clean common LLM SQL formatting artifacts."""

        sql = text.strip()

        if sql.startswith("```"):
            lines = sql.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            sql = "\n".join(lines).strip()

        if sql.lower().startswith("sql\n"):
            sql = sql[4:].strip()

        return sql