"""Startup and import verification — confirms the scaffold is functional."""

import importlib
import pytest
from app.config import AppSettings, get_settings, reset_settings
from gateway.llm_provider import get_llm
from gateway.mock_llm import MOCK_PREFIX


PACKAGES = ["app", "agents", "tools", "ml", "rag", "gateway", "eval", "tests", "ui"]


@pytest.fixture(autouse=True)
def _clean_settings():
    reset_settings()
    yield
    reset_settings()


class TestPackageImports:

    @pytest.mark.parametrize("package", PACKAGES)
    def test_package_importable(self, package):
        mod = importlib.import_module(package)
        assert mod is not None


class TestAppConfig:

    def test_settings_load_defaults(self):
        s = AppSettings(
            llm_provider="mock", llm_model="mock-response", mock_llm=True,
        )
        assert s.app_env == "development"
        assert s.log_level == "INFO"
        assert s.llm_provider == "mock"

    def test_secret_not_in_repr(self):
        s = AppSettings(api_secret_key="hunter2", gemini_api_key="abc123")
        r = repr(s)
        assert "hunter2" not in r
        assert "abc123" not in r

    def test_singleton_returns_same_instance(self):
        a = get_settings()
        b = get_settings()
        assert a is b


class TestLLMProviderFactory:

    def test_mock_provider_returns_mock(self):
        llm = get_llm()
        resp = llm.invoke("hello")
        assert resp.mock is True
        assert resp.content.startswith(MOCK_PREFIX)

    def test_mock_no_network_calls(self):
        llm = get_llm()
        for _ in range(10):
            llm.invoke("test")
        assert llm.call_count == 10


class TestCoreImports:
    """Verify that installed dependencies are importable."""

    @pytest.mark.parametrize("module", [
        "pydantic",
        "pydantic_settings",
        "dotenv",
        "sqlalchemy",
        "fastapi",
        "uvicorn",
        "litellm",
    ])
    def test_dependency_importable(self, module):
        importlib.import_module(module)
