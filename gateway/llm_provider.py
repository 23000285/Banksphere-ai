"""LLM provider factory — returns mock or LiteLLM-backed client.

Usage:
    from gateway.llm_provider import get_llm
    llm = get_llm()
    response = llm.invoke("What is my balance?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from gateway.llm_config import LLMConfig, load_llm_config
from gateway.mock_llm import MockLLM, MockLLMResponse

logger = logging.getLogger(__name__)


class LLMResponse(Protocol):
    content: str
    model: str
    mock: bool


@dataclass
class LiteLLMResponse:
    content: str
    model: str
    mock: bool = False
    trace_id: str = ""


@dataclass
class LiteLLMWrapper:
    """Thin wrapper around litellm.completion for a consistent interface."""

    config: LLMConfig
    _call_count: int = field(default=0, repr=False)

    def invoke(self, prompt: str) -> LiteLLMResponse:
        import litellm

        self._call_count += 1

        kwargs: dict = {
            "model": self.config.litellm_model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.config.provider == "gemini" and self.config.gemini_api_key:
            kwargs["api_key"] = self.config.gemini_api_key

        if self.config.provider == "bedrock" and self.config.aws_region:
            kwargs["aws_region_name"] = self.config.aws_region

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""

        return LiteLLMResponse(
            content=content,
            model=self.config.litellm_model,
            trace_id=response.id if hasattr(response, "id") else "",
        )

    @property
    def call_count(self) -> int:
        return self._call_count


def get_llm(config: LLMConfig | None = None) -> MockLLM | LiteLLMWrapper:
    """Return the appropriate LLM client based on configuration.

    In mock mode (the default), returns a MockLLM that needs no network or keys.
    Otherwise returns a LiteLLMWrapper for the configured provider.
    """
    if config is None:
        config = load_llm_config()

    if config.is_mock:
        logger.info("Using mock LLM provider (no API calls will be made)")
        return MockLLM(model=config.model)

    logger.info("Using %s LLM provider (model: %s)", config.provider, config.litellm_model)
    return LiteLLMWrapper(config=config)
