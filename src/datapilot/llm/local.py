from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_lm import generate, load


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for the local DataPilot LLM."""

    model_name: str = "Qwen/Qwen3-8B-MLX-8bit"
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    """Structured response returned by the local LLM."""

    text: str
    model_name: str


class LocalLLM:
    """Local LLM runtime backed by MLX-LM."""

    def __init__(self, config: LocalLLMConfig | None = None) -> None:
        self.config = config or LocalLLMConfig()

        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load_model(self) -> None:
        """Load the configured model into memory."""

        if self._model is not None and self._tokenizer is not None:
            return

        self._model, self._tokenizer = load(self.config.model_name)

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response for a single user prompt."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        self.load_model()

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.config.max_tokens,
            verbose=False,
        )

        return LLMResponse(
            text=response.strip(),
            model_name=self.config.model_name,
        )