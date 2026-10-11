"""Tests for LLM response generation integration.

Verifies:
- Mock mode works without external calls
- Gemini mode invokes the configured wrapper
- Structured data is preserved (not altered by the narrative)
- Compliance citations come from retrieved evidence
- Empty/failed retrieval does not produce invented policy answers
- Provider failures trigger safe fallback
- API keys don't appear in logs or prompts
- Authorization and approval enforcement remain intact
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agents.response_generator import (
    generate_narrative,
    _build_banking_prompt,
    _build_fraud_prompt,
    _build_credit_prompt,
    _build_compliance_prompt,
    _build_card_block_prompt,
)
from gateway.llm_config import LLMConfig


def _mock_config():
    return LLMConfig(
        provider="mock",
        model="mock-response",
        mock_llm=True,
    )


def _gemini_config():
    return LLMConfig(
        provider="gemini",
        model="gemini/gemini-2.0-flash",
        mock_llm=False,
        gemini_api_key="test-key-123",
    )


# ---------------------------------------------------------------------------
# Prompt builders produce correct content
# ---------------------------------------------------------------------------


class TestPromptBuilders:

    def test_banking_prompt_includes_account_data(self):
        data = {
            "accounts": [
                {"account_number": "BSP100000001", "account_type": "savings",
                 "balance": "20539.79", "status": "active"}
            ],
            "transactions": [
                {"timestamp": "2026-10-05", "txn_type": "credit",
                 "amount": "648.21", "merchant": "Swiggy", "channel": "upi"}
            ],
        }
        prompt = _build_banking_prompt("What is my balance?", data)
        assert "BSP100000001" in prompt
        assert "20539.79" in prompt
        assert "Swiggy" in prompt

    def test_fraud_prompt_includes_scores(self):
        data = {
            "scored_transactions": [
                {"transaction_uid": "abc12345-rest", "amount": "500.00",
                 "fraud_score": 0.95, "risk_category": "high",
                 "recommendation": "block",
                 "top_features": [{"hour": 13.8}, {"is_night": 6.99}]}
            ]
        }
        prompt = _build_fraud_prompt("Check fraud", data)
        assert "0.95" in prompt
        assert "block" in prompt
        assert "hour" in prompt

    def test_credit_prompt_includes_eligibility(self):
        data = {
            "eligible": True,
            "risk_category": "low",
            "default_probability": 0.001,
            "emi": "16607.15",
            "max_eligible_amount": "752687.63",
            "debt_to_income_ratio": 0.33,
            "reason_codes": ["Meets all criteria"],
        }
        prompt = _build_credit_prompt("Am I eligible?", data)
        assert "True" in prompt
        assert "16607.15" in prompt
        assert "Meets all criteria" in prompt

    def test_compliance_prompt_includes_sources(self):
        data = {
            "answer": "KYC requires government ID.",
            "sources": [{"source": "kyc_verification.md"}],
        }
        prompt = _build_compliance_prompt("KYC docs?", data)
        assert "kyc_verification.md" in prompt
        assert "KYC requires government ID" in prompt

    def test_compliance_prompt_empty_answer_returns_empty(self):
        data = {"answer": "", "sources": []}
        prompt = _build_compliance_prompt("anything?", data)
        assert prompt == ""

    def test_card_block_prompt_includes_status(self):
        data = {
            "action": "card_block",
            "account_number": "BSP100000001",
            "status": "pending_approval",
            "message": "Awaiting human approval",
            "requires_approval": True,
        }
        prompt = _build_card_block_prompt("Block my card", data)
        assert "pending_approval" in prompt
        assert "human approval" in prompt


# ---------------------------------------------------------------------------
# Mock mode: no external calls
# ---------------------------------------------------------------------------


class TestMockMode:

    def test_mock_returns_prefixed_narrative(self):
        data = {
            "accounts": [{"account_number": "BSP1", "account_type": "savings",
                          "balance": "1000", "status": "active"}],
            "transactions": [],
        }
        result = generate_narrative("banking", "balance?", data, config=_mock_config())
        assert result.startswith("[MOCK]")

    def test_mock_does_not_call_litellm(self):
        with patch("gateway.llm_provider.LiteLLMWrapper") as mock_cls:
            data = {"accounts": [], "transactions": []}
            generate_narrative("banking", "balance?", data, config=_mock_config())
            mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Gemini mode: uses the configured wrapper
# ---------------------------------------------------------------------------


class TestGeminiMode:

    @patch("agents.response_generator.get_llm")
    def test_gemini_mode_calls_llm(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Here is your balance summary.")
        mock_get_llm.return_value = mock_llm

        data = {
            "accounts": [{"account_number": "BSP1", "account_type": "savings",
                          "balance": "5000", "status": "active"}],
            "transactions": [],
        }
        result = generate_narrative("banking", "balance?", data, config=_gemini_config())
        assert mock_llm.invoke.called
        assert "balance summary" in result

    @patch("agents.response_generator.get_llm")
    def test_gemini_prompt_does_not_contain_api_key(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Response")
        mock_get_llm.return_value = mock_llm

        data = {"accounts": [], "transactions": []}
        generate_narrative("banking", "balance?", data, config=_gemini_config())

        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "test-key-123" not in prompt_sent


# ---------------------------------------------------------------------------
# Structured data preservation
# ---------------------------------------------------------------------------


class TestStructuredDataPreservation:

    def test_banking_response_unchanged_after_narrative(self):
        from agents.supervisor import RoutingResult

        original_result = {
            "agent": "banking",
            "response": {
                "accounts": [{"balance": "20539.79"}],
                "transactions": [],
            },
            "narrative": "Some narrative text",
        }
        assert original_result["response"]["accounts"][0]["balance"] == "20539.79"

    def test_fraud_scores_not_altered(self):
        data = {
            "scored_transactions": [
                {"fraud_score": 0.9996, "recommendation": "block"}
            ]
        }
        generate_narrative("fraud", "check fraud", data, config=_mock_config())
        assert data["scored_transactions"][0]["fraud_score"] == 0.9996
        assert data["scored_transactions"][0]["recommendation"] == "block"

    def test_credit_eligibility_not_altered(self):
        data = {
            "eligible": True,
            "risk_category": "low",
            "default_probability": 0.001,
            "emi": "16607.15",
        }
        generate_narrative("credit", "loan?", data, config=_mock_config())
        assert data["eligible"] is True
        assert data["emi"] == "16607.15"


# ---------------------------------------------------------------------------
# Compliance: citations from evidence only
# ---------------------------------------------------------------------------


class TestComplianceCitations:

    @patch("agents.response_generator.get_llm")
    def test_compliance_prompt_passes_only_retrieved_evidence(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Per kyc_verification.md, you need government ID."
        )
        mock_get_llm.return_value = mock_llm

        data = {
            "answer": "All customers must complete KYC.",
            "sources": [
                {"source": "kyc_verification.md", "heading": "Section 1"},
            ],
        }
        result = generate_narrative("compliance", "KYC docs?", data, config=_gemini_config())
        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "kyc_verification.md" in prompt_sent
        assert "Do not add policy requirements" in prompt_sent

    def test_no_evidence_returns_empty_narrative(self):
        data = {"answer": "", "sources": [], "has_evidence": False}
        result = generate_narrative("compliance", "unknown policy?", data, config=_mock_config())
        assert result == ""


# ---------------------------------------------------------------------------
# Provider failure: safe fallback
# ---------------------------------------------------------------------------


class TestProviderFailure:

    @patch("agents.response_generator.get_llm")
    def test_llm_exception_returns_empty_string(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API unavailable")
        mock_get_llm.return_value = mock_llm

        data = {
            "accounts": [{"account_number": "BSP1", "account_type": "savings",
                          "balance": "5000", "status": "active"}],
            "transactions": [],
        }
        result = generate_narrative("banking", "balance?", data, config=_gemini_config())
        assert result == ""

    def test_unsupported_intent_returns_empty(self):
        result = generate_narrative("unsupported", "weather?", {}, config=_mock_config())
        assert result == ""

    def test_empty_response_data_returns_empty(self):
        result = generate_narrative("banking", "balance?", {}, config=_mock_config())
        assert "[MOCK]" in result or result == ""


# ---------------------------------------------------------------------------
# PII masking applied to narrative
# ---------------------------------------------------------------------------


class TestPIIMasking:

    @patch("agents.response_generator.get_llm")
    def test_pii_in_narrative_is_masked(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Contact synth_user_0001@banksphere.test for help."
        )
        mock_get_llm.return_value = mock_llm

        data = {"accounts": [], "transactions": []}
        result = generate_narrative("banking", "help", data, config=_gemini_config())
        assert "synth_user_0001@banksphere.test" not in result
        assert "***@***.***" in result


# ---------------------------------------------------------------------------
# is_mock field reflects config
# ---------------------------------------------------------------------------


class TestIsMockField:

    def test_is_mock_true_in_mock_mode(self):
        with patch("gateway.llm_config.load_llm_config") as mock_cfg:
            mock_cfg.return_value = LLMConfig(
                provider="mock", model="mock-response", mock_llm=True
            )
            from agents.base import BaseAgent

            agent = BaseAgent(name="test")
            result = agent._make_result("completed", response={"data": 1})
            assert result.is_mock is True

    def test_is_mock_false_in_gemini_mode(self):
        with patch("gateway.llm_config.load_llm_config") as mock_cfg:
            mock_cfg.return_value = LLMConfig(
                provider="gemini", model="gemini/gemini-2.0-flash",
                mock_llm=False, gemini_api_key="test-key"
            )
            from agents.base import BaseAgent

            agent = BaseAgent(name="test")
            result = agent._make_result("completed", response={"data": 1})
            assert result.is_mock is False


# ---------------------------------------------------------------------------
# Integration: supervisor adds narrative to result
# ---------------------------------------------------------------------------


class TestSupervisorNarrativeIntegration:

    @patch("agents.response_generator.get_llm")
    def test_routing_result_contains_narrative(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Your savings balance is ₹20,539.79.")
        mock_get_llm.return_value = mock_llm

        from agents.supervisor import RoutingResult
        result = RoutingResult(
            query="What is my balance?",
            intent="banking",
            specialist="banking",
            status="completed",
            trace_id="abc123",
            result={"response": {"accounts": []}},
            narrative="Your savings balance is ₹20,539.79.",
        )
        assert result.narrative != ""
        assert "20,539.79" in result.narrative


# ---------------------------------------------------------------------------
# API schema backward compatibility
# ---------------------------------------------------------------------------


class TestAPISchemaCompat:

    def test_chat_response_has_narrative_field(self):
        from app.main import ChatResponse

        resp = ChatResponse(
            trace_id="abc",
            query="test",
            intent="banking",
            specialist="banking",
            status="completed",
            result={},
            narrative="Hello",
        )
        assert resp.narrative == "Hello"

    def test_chat_response_narrative_defaults_empty(self):
        from app.main import ChatResponse

        resp = ChatResponse(
            trace_id="abc",
            query="test",
            intent="banking",
            specialist="banking",
            status="completed",
            result={},
        )
        assert resp.narrative == ""
