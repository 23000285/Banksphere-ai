"""Tests for LLM configuration loading, validation, and mock provider."""

import pytest
from app.config import AppSettings, reset_settings
from gateway.llm_config import LLMConfig, LLMConfigError, load_llm_config
from gateway.mock_llm import MockLLM, MOCK_PREFIX


@pytest.fixture(autouse=True)
def _clean_settings():
    """Reset cached settings before each test."""
    reset_settings()
    yield
    reset_settings()


def _make_settings(**overrides) -> AppSettings:
    """Build an AppSettings with test defaults, without reading .env."""
    defaults = {
        "llm_provider": "mock",
        "llm_model": "mock-response",
        "mock_llm": True,
        "gemini_api_key": "",
        "aws_region": "",
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

class TestLoadLLMConfig:

    def test_defaults_to_mock(self):
        cfg = load_llm_config(_make_settings())
        assert cfg.provider == "mock"
        assert cfg.mock_llm is True
        assert cfg.is_mock is True

    def test_explicit_mock(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="mock", llm_model="mock-response", mock_llm=True))
        assert cfg.provider == "mock"
        assert cfg.is_mock is True
        assert cfg.litellm_model == ""

    def test_mock_provider_requires_mock_flag(self):
        with pytest.raises(LLMConfigError, match="MOCK_LLM=true"):
            load_llm_config(_make_settings(llm_provider="mock", mock_llm=False))

    def test_invalid_provider(self):
        with pytest.raises(LLMConfigError, match="not supported"):
            load_llm_config(_make_settings(llm_provider="openai", mock_llm=False))

    def test_gemini_requires_api_key(self):
        with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
            load_llm_config(_make_settings(
                llm_provider="gemini", llm_model="gemini-2.0-flash", mock_llm=False))

    def test_gemini_with_key(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="gemini", llm_model="gemini-2.0-flash",
            mock_llm=False, gemini_api_key="test-key-123"))
        assert cfg.provider == "gemini"
        assert cfg.is_mock is False
        assert cfg.litellm_model == "gemini/gemini-2.0-flash"

    def test_gemini_model_prefix_preserved(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="gemini", llm_model="gemini/gemini-2.0-flash",
            mock_llm=False, gemini_api_key="test-key-123"))
        assert cfg.litellm_model == "gemini/gemini-2.0-flash"

    def test_gemini_with_mock_override(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="gemini", llm_model="gemini-2.0-flash", mock_llm=True))
        assert cfg.is_mock is True

    def test_bedrock_requires_region(self):
        with pytest.raises(LLMConfigError, match="AWS_REGION"):
            load_llm_config(_make_settings(
                llm_provider="bedrock",
                llm_model="anthropic.claude-3-sonnet-20240229-v1:0",
                mock_llm=False))

    def test_bedrock_with_region(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="bedrock",
            llm_model="anthropic.claude-3-sonnet-20240229-v1:0",
            mock_llm=False, aws_region="us-east-1"))
        assert cfg.provider == "bedrock"
        assert cfg.is_mock is False

    def test_api_key_not_in_repr(self):
        cfg = load_llm_config(_make_settings(
            llm_provider="gemini", llm_model="gemini-2.0-flash",
            mock_llm=False, gemini_api_key="secret-key-xyz"))
        assert "secret-key-xyz" not in repr(cfg)


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class TestMockLLM:

    def test_response_is_prefixed(self):
        llm = MockLLM()
        resp = llm.invoke("What is my balance?")
        assert resp.content.startswith(MOCK_PREFIX)

    def test_response_is_flagged_as_mock(self):
        llm = MockLLM()
        resp = llm.invoke("Check my balance")
        assert resp.mock is True
        assert resp.model == "mock-response"

    def test_deterministic_same_input(self):
        llm = MockLLM()
        r1 = llm.invoke("Show my balance")
        r2 = llm.invoke("Show my balance")
        assert r1.content == r2.content
        assert r1.trace_id == r2.trace_id

    def test_intent_matching_balance(self):
        llm = MockLLM()
        resp = llm.invoke("What is my account balance?")
        assert "balance" in resp.content.lower()

    def test_intent_matching_fraud(self):
        llm = MockLLM()
        resp = llm.invoke("Is this transaction fraud?")
        assert "risk" in resp.content.lower()

    def test_intent_matching_credit(self):
        llm = MockLLM()
        resp = llm.invoke("Check my credit score")
        assert "credit" in resp.content.lower()

    def test_intent_matching_kyc(self):
        llm = MockLLM()
        resp = llm.invoke("What are the KYC requirements?")
        assert "kyc" in resp.content.lower()

    def test_intent_matching_card(self):
        llm = MockLLM()
        resp = llm.invoke("Block my card immediately")
        assert "card" in resp.content.lower()

    def test_unknown_intent_returns_default(self):
        llm = MockLLM()
        resp = llm.invoke("Tell me about the weather")
        assert MOCK_PREFIX in resp.content
        assert "simulated" in resp.content.lower()

    def test_call_count_increments(self):
        llm = MockLLM()
        assert llm.call_count == 0
        llm.invoke("hello")
        llm.invoke("world")
        assert llm.call_count == 2

    def test_trace_id_is_stable(self):
        llm = MockLLM()
        r1 = llm.invoke("test prompt")
        r2 = llm.invoke("test prompt")
        assert r1.trace_id == r2.trace_id
        assert len(r1.trace_id) == 12

    def test_different_prompts_different_traces(self):
        llm = MockLLM()
        r1 = llm.invoke("prompt one")
        r2 = llm.invoke("prompt two")
        assert r1.trace_id != r2.trace_id
