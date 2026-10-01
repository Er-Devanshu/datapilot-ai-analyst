# LLM Layer

The `llm` package contains the language-model integration used by DataPilot.

This layer provides the interface between the agent pipeline and the language model responsible for tasks such as SQL generation and grounded answer composition.

## Responsibilities

The LLM layer is responsible for:

- Providing a consistent interface for language-model interactions.
- Encapsulating model-specific implementation details.
- Generating text from prompts supplied by the agent pipeline.
- Keeping LLM integration separate from agent orchestration and business logic.
- Supporting deterministic test doubles and alternative model implementations.

## Role in DataPilot

The LLM layer is consumed by higher-level components rather than controlling the overall agent workflow.

```text
User Question
      │
      ▼
 Agent Pipeline
      │
      ├──────────────► SQL Generation
      │                     │
      │                     ▼
      │                    LLM
      │
      └──────────────► Answer Composition
                            │
                            ▼
                           LLM