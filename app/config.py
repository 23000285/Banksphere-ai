"""Application-wide settings loaded from environment variables and .env file."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/banksphere.db"
    api_secret_key: str = Field(default="CHANGE_ME_BEFORE_USE", repr=False)

    llm_provider: str = "mock"
    llm_model: str = "mock-response"
    mock_llm: bool = True
    gemini_api_key: str = Field(default="", repr=False)
    aws_region: str = ""


_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Return cached application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reset_settings() -> None:
    """Clear cached settings — used by tests."""
    global _settings
    _settings = None
