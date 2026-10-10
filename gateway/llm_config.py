"""Centralized LLM configuration — loads settings from environment variables.

Supports three providers:
  mock    — deterministic responses, no API key, no network (default)
  gemini  — Google Gemini via LiteLLM, requires GEMINI_API_KEY
  bedrock — Amazon Bedrock via LiteLLM, requires AWS credentials
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

VALID_PROVIDERS = ("mock", "gemini", "bedrock")

ProviderName = Literal["mock", "gemini", "bedrock"]


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid or incomplete."""


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName
    model: str
    mock_llm: bool
    gemini_api_key: str = field(default="", repr=False)
    aws_region: str = ""

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock" or self.mock_llm

    @property
    def litellm_model(self) -> str:
        """Return the model string LiteLLM expects, or empty for mock."""
        if self.is_mock:
            return ""
        if self.provider == "gemini":
            return self.model if self.model.startswith("gemini/") else f"gemini/{self.model}"
        return self.model


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def load_llm_config() -> LLMConfig:
    """Load and validate LLM configuration from environment variables.

    Returns a validated LLMConfig. Raises LLMConfigError with an actionable
    message when required settings are missing or inconsistent.
    """
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    model = os.getenv("LLM_MODEL", "mock-response").strip()
    mock_llm = _parse_bool(os.getenv("MOCK_LLM", "true"))
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    aws_region = os.getenv("AWS_REGION", "")

    if provider not in VALID_PROVIDERS:
        raise LLMConfigError(
            f"LLM_PROVIDER={provider!r} is not supported. "
            f"Valid providers: {', '.join(VALID_PROVIDERS)}"
        )

    if provider == "mock" and not mock_llm:
        raise LLMConfigError(
            "LLM_PROVIDER=mock requires MOCK_LLM=true. "
            "Set MOCK_LLM=true or choose a different provider."
        )

    if provider == "gemini" and not mock_llm:
        if not gemini_api_key:
            raise LLMConfigError(
                "LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set. "
                "Add it to your .env file (never commit the key). "
                "Get a free key at https://aistudio.google.com/apikey"
            )

    if provider == "bedrock" and not mock_llm:
        if not aws_region:
            raise LLMConfigError(
                "LLM_PROVIDER=bedrock requires AWS_REGION to be set. "
                "Also ensure AWS credentials are configured."
            )

    return LLMConfig(
        provider=provider,
        model=model,
        mock_llm=mock_llm,
        gemini_api_key=gemini_api_key,
        aws_region=aws_region,
    )
