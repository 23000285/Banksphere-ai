"""Tests for LLM configuration loading, validation, and mock provider."""

import os
import pytest
from gateway.llm_config import LLMConfig, LLMConfigError, load_llm_config
from gateway.mock_llm import MockLLM, MockLLMResponse, MOCK_PREFIX


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

class TestLoadLLMConfig:
    """Test load_llm_config with various environment states."""

    def _set_env(self, monkeypatch, **kwargs):
        for key in ("LLM_PROVIDER", "LLM_MODEL", "MOCK_LLM",
                     "GEMINI_API_KEY", "AWS_REGION"):
            monkeypatch.delenv(key, raising=False)
        for key, val in kwargs.items():
            monkeypatch.setenv(key, val)

    def test_defaults_to_mock(self, monkeypatch):
        self._set_env(monkeypatch)
        cfg = load_llm_config()
        assert cfg.provider == "mock"
        assert cfg.mock_llm is True
        assert cfg.is_mock is True

    def test_explicit_mock(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="mock", LLM_MODEL="mock-response", MOCK_LLM="true")
        cfg = load_llm_config()
        assert cfg.provider == "mock"
        assert cfg.is_mock is True
        assert cfg.litellm_model == ""

    def test_mock_provider_requires_mock_flag(self, monkeypatch):
        self._set_env(monkeypatch, LLM_PROVIDER="mock", MOCK_LLM="false")
        with pytest.raises(LLMConfigError, match="MOCK_LLM=true"):
            load_llm_config()

    def test_invalid_provider(self, monkeypatch):
        self._set_env(monkeypatch, LLM_PROVIDER="openai", MOCK_LLM="false")
        with pytest.raises(LLMConfigError, match="not supported"):
            load_llm_config()

    def test_gemini_requires_api_key(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="gemini", LLM_MODEL="gemini-2.0-flash", MOCK_LLM="false")
        with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
            load_llm_config()

    def test_gemini_with_key(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="gemini", LLM_MODEL="gemini-2.0-flash",
                      MOCK_LLM="false", GEMINI_API_KEY="test-key-123")
        cfg = load_llm_config()
        assert cfg.provider == "gemini"
        assert cfg.is_mock is False
        assert cfg.litellm_model == "gemini/gemini-2.0-flash"

    def test_gemini_model_prefix_preserved(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="gemini", LLM_MODEL="gemini/gemini-2.0-flash",
                      MOCK_LLM="false", GEMINI_API_KEY="test-key-123")
        cfg = load_llm_config()
        assert cfg.litellm_model == "gemini/gemini-2.0-flash"

    def test_gemini_with_mock_override(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="gemini", LLM_MODEL="gemini-2.0-flash", MOCK_LLM="true")
        cfg = load_llm_config()
        assert cfg.is_mock is True

    def test_bedrock_requires_region(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="bedrock",
                      LLM_MODEL="anthropic.claude-3-sonnet-20240229-v1:0",
                      MOCK_LLM="false")
        with pytest.raises(LLMConfigError, match="AWS_REGION"):
            load_llm_config()

    def test_bedrock_with_region(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="bedrock",
                      LLM_MODEL="anthropic.claude-3-sonnet-20240229-v1:0",
                      MOCK_LLM="false", AWS_REGION="us-east-1")
        cfg = load_llm_config()
        assert cfg.provider == "bedrock"
        assert cfg.is_mock is False

    def test_api_key_not_in_repr(self, monkeypatch):
        self._set_env(monkeypatch,
                      LLM_PROVIDER="gemini", LLM_MODEL="gemini-2.0-flash",
                      MOCK_LLM="false", GEMINI_API_KEY="secret-key-xyz")
        cfg = load_llm_config()
        assert "secret-key-xyz" not in repr(cfg)


# ---------------------------------------------------------------------------
# Mock LLM provider
# ---------------------------------------------------------------------------

class TestMockLLM:
    """Test the deterministic mock LLM."""

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
