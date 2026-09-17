from __future__ import annotations

from dataclasses import dataclass

from datapilot.llm.local import LocalLLM
from datapilot.llm.prompts import SQLPromptBuilder


@dataclass(frozen=True)
class SQLGenerationResult:
    """Result returned by the SQL generator."""

    question: str
    sql: str
    model_name: str


class SQLGenerator:
    """Generate SQL from natural-language questions."""

    def __init__(
        self,
        llm: LocalLLM,
        prompt_builder: SQLPromptBuilder | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = (
            prompt_builder
            or SQLPromptBuilder()
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

        sql = self._clean_sql(response.text)

        if not sql:
            raise RuntimeError(
                "The LLM returned an empty SQL query."
            )

        return SQLGenerationResult(
            question=question,
            sql=sql,
            model_name=response.model_name,
        )

    @staticmethod
    def _clean_sql(text: str) -> str:
        """Clean common formatting artifacts from LLM output."""

        sql = text.strip()

        if sql.startswith("```sql"):
            sql = sql[6:]

        elif sql.startswith("```"):
            sql = sql[3:]

        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()