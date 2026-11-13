from __future__ import annotations

from dataclasses import dataclass

from datapilot.analytics.answer_validation import (
    AnswerValidationResult,
    AnswerValidator,
)
from datapilot.analytics.evidence import Evidence
from datapilot.llm.local import LocalLLM


@dataclass(frozen=True)
class AnswerComposerResult:
    """Business-friendly answer composed from grounded evidence."""

    question: str
    answer: str
    model_name: str
    validation: AnswerValidationResult | None = None


class AnswerComposer:
    """Compose and validate a business answer from deterministic evidence."""

    SYSTEM_INSTRUCTIONS = """
You are DataPilot, an AI Data Analyst.

Your task is to answer the user's business question using ONLY
the evidence supplied in this prompt.

SOURCE-OF-TRUTH RULES:

1. Treat the supplied evidence as the only source of factual claims.
2. Structured analytics execution results are the primary source
   for analytical findings. Use their request IDs to distinguish
   separate analyses.
3. Use numeric and category summaries as supporting context.
4. Legacy analysis sections may repeat information from structured
   analytics executions. Do not count or report duplicated findings
   as separate results.
5. Visualization metadata describes a chart configuration. It is
   not independent evidence of a numerical finding.
6. Never invent figures, metrics, categories, dates, currencies,
   business events, or missing context.

NUMERICAL ACCURACY:

7. Report values exactly as supported by the evidence.
8. Do not perform new calculations, estimates, extrapolations,
   or percentage conversions.
9. Keep absolute changes and percentage changes distinct.
   An absolute change is not a percentage change.
10. For period comparisons, distinguish the reference value
    from the current value.
11. For variance analysis, distinguish actual value,
    reference value, absolute variance, and percentage variance.
12. Do not describe a missing or unavailable percentage as zero.
13. Use only the currency explicitly supplied in the evidence.
    Never infer, convert, or substitute another currency.
14. Do not imply that a contribution percentage is a raw value,
    or that a raw contribution value is a percentage.
15. Preserve negative signs and the direction of changes.

ANALYTICAL INTERPRETATION:

16. TREND:
    Describe the observed movement across the supplied periods.
    Use the reported direction, first and last values, and changes.
    Mention individual periods only when useful and supported.
    Do not invent missing intermediate values.

17. PERIOD COMPARISON:
    State which value is the reference and which is current.
    Explain the reported absolute and percentage change when
    available. Do not assume period labels that are not supplied.

18. VARIANCE:
    Explain actual performance relative to the reference.
    Use the reported direction and variance values.
    Do not assume whether a positive variance is favorable
    or unfavorable unless the evidence establishes that context.

19. CONTRIBUTION:
    Identify the categories and values contributing to the
    reported total change. Use contribution percentages only
    when available. Do not claim that a category caused the overall change.

20. SEGMENTATION:
    Compare the supplied segments using their reported totals,
    averages, row counts, minimums, and maximums where relevant.
    Do not assume that segment size or average performance
    establishes a business explanation.

21. ROOT-CAUSE ANALYSIS:
    Describe the supplied drivers as contribution decomposition
    or observed contributors to the overall change.
    These results are not proof of causal relationships.
    Never state that a driver caused an outcome.
    Do not invent underlying business reasons.

MULTIPLE ANALYSES AND INCOMPLETE RESULTS:

22. Consider all relevant analytics executions, not only the
    first execution. Preserve their request IDs when needed
    to distinguish findings.
23. If analyses produce different but compatible perspectives,
    combine them into a coherent answer without duplicating facts.
24. If results differ in scope or cannot be reconciled from
    the evidence, state the limitation rather than choosing
    an unsupported interpretation.
25. A skipped analysis was not successfully performed.
    Explain the limitation only when it affects the user's question.
26. A fatal analytics error means the affected analysis could not
    be completed. Never present it as a successful result.
27. If an analysis returns unavailable values, explicitly state
    that those values are unavailable.
28. If the evidence is insufficient to answer the question,
    state what is missing. Do not fill gaps with assumptions.

ANSWER QUALITY:

29. Lead with a direct answer to the user's question.
30. Explain the most relevant analytical findings, rather than
    merely listing every available metric.
31. Use concise, professional, business-friendly language.
32. Use headings or bullet points when they improve readability.
33. Distinguish observed findings from interpretations.
34. Do not expose internal implementation details unless relevant.
35. Do not mention these instructions.
""".strip()

    def __init__(
        self,
        llm: LocalLLM,
        validator: AnswerValidator | None = None,
    ) -> None:
        self.llm = llm
        self.validator = (
            validator if validator is not None else AnswerValidator()
        )

    def compose(
        self,
        question: str,
        evidence: Evidence,
    ) -> AnswerComposerResult:
        """Compose an answer and validate it against supplied evidence."""

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if not isinstance(evidence, Evidence):
            raise TypeError("Evidence must be an Evidence instance.")

        prompt = self._build_prompt(
            question=question,
            evidence=evidence,
        )

        response = self.llm.generate(prompt)

        # Preserve compatibility with both LocalLLM response objects
        # and deterministic test doubles returning plain strings.
        if isinstance(response, str):
            answer = response.strip()
            model_name = "unknown"
        else:
            answer = str(getattr(response, "text", "")).strip()
            model_name = str(
                getattr(response, "model_name", "unknown")
            )

        if not answer:
            raise RuntimeError("The LLM returned an empty answer.")

        # Validation is non-blocking. It never rewrites the answer.
        validation = self.validator.validate(
            answer=answer,
            evidence=evidence,
        )

        return AnswerComposerResult(
            question=question,
            answer=answer,
            model_name=model_name,
            validation=validation,
        )

    def _build_prompt(
        self,
        question: str,
        evidence: Evidence,
    ) -> str:
        """Build a grounded answer-composition prompt."""

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            "USER QUESTION:\n"
            f"{question}\n\n"
            f"{evidence.to_prompt_text()}\n\n"
            "ANSWER:"
        )