"""Tests for Streamlit UI helpers, formatters, and importability."""

from __future__ import annotations


class TestFormatHelpers:
    """Test response formatting functions without running Streamlit."""

    def test_format_banking_accounts(self) -> None:
        from ui.streamlit_app import _format_banking
        resp = {
            "accounts": [
                {"account_type": "savings", "account_number": "ACC001",
                 "balance": "150000.50", "currency": "INR", "status": "active"},
            ],
            "transactions": [],
        }
        text = _format_banking(resp)
        assert "Account Summary" in text
        assert "savings" in text
        assert "ACC001" in text
        assert "150,000.50" in text

    def test_format_banking_with_transactions(self) -> None:
        from ui.streamlit_app import _format_banking
        resp = {
            "accounts": [],
            "transactions": [
                {"timestamp": "2025-01-15T10:00:00", "txn_type": "debit",
                 "category": "food", "amount": "500.00", "merchant": "Cafe",
                 "is_flagged": False},
            ],
        }
        text = _format_banking(resp)
        assert "Recent Transactions" in text
        assert "Cafe" in text

    def test_format_banking_flagged_transaction(self) -> None:
        from ui.streamlit_app import _format_banking
        resp = {
            "accounts": [],
            "transactions": [
                {"timestamp": "2025-01-15", "txn_type": "debit",
                 "category": "other", "amount": "99999", "merchant": "Unknown",
                 "is_flagged": True},
            ],
        }
        text = _format_banking(resp)
        assert "\U0001f6a9" in text

    def test_format_banking_empty(self) -> None:
        from ui.streamlit_app import _format_banking
        text = _format_banking({"accounts": [], "transactions": []})
        assert "Account Summary" in text

    def test_format_fraud_scored(self) -> None:
        from ui.streamlit_app import _format_fraud
        resp = {
            "scored_transactions": [
                {"transaction_uid": "txn-abcdef12", "amount": "5000",
                 "fraud_score": 0.85, "risk_category": "high",
                 "recommendation": "block", "requires_approval": True},
                {"transaction_uid": "txn-00000000", "amount": "200",
                 "fraud_score": 0.01, "risk_category": "low",
                 "recommendation": "allow", "requires_approval": False},
            ],
        }
        text = _format_fraud(resp)
        assert "Fraud Analysis" in text
        assert "2 transactions scored" in text
        assert "\U0001f534" in text
        assert "\U0001f7e2" in text
        assert "Requires human approval" in text

    def test_format_fraud_empty(self) -> None:
        from ui.streamlit_app import _format_fraud
        text = _format_fraud({"scored_transactions": []})
        assert "No transactions scored" in text

    def test_format_credit_eligible(self) -> None:
        from ui.streamlit_app import _format_credit
        resp = {
            "eligible": True,
            "risk_category": "low",
            "default_probability": 0.05,
            "monthly_emi": 15000.0,
            "total_payment": 540000.0,
            "total_interest": 40000.0,
        }
        text = _format_credit(resp)
        assert "Loan Eligibility" in text
        assert "✅" in text
        assert "Yes" in text
        assert "0.0500" in text

    def test_format_credit_ineligible(self) -> None:
        from ui.streamlit_app import _format_credit
        resp = {"eligible": False, "reason_codes": ["High DTI ratio", "Low credit score"]}
        text = _format_credit(resp)
        assert "❌" in text
        assert "High DTI ratio" in text

    def test_format_compliance_with_sources(self) -> None:
        from ui.streamlit_app import _format_compliance
        resp = {
            "answer": "KYC verification requires government-issued photo ID.",
            "sources": [
                {"source": "kyc_verification.md", "category": "KYC", "heading": "Requirements"},
                {"source": "kyc_verification.md", "category": "KYC", "heading": "Documents"},
            ],
        }
        text = _format_compliance(resp)
        assert "KYC verification" in text
        assert "Sources:" in text
        assert "kyc_verification.md" in text
        assert "Requirements" in text

    def test_format_compliance_without_sources(self) -> None:
        from ui.streamlit_app import _format_compliance
        resp = {"answer": "Some answer", "sources": []}
        text = _format_compliance(resp)
        assert "Some answer" in text
        assert "Sources:" not in text

    def test_format_compliance_no_evidence(self) -> None:
        from ui.streamlit_app import _format_compliance
        text = _format_compliance({})
        assert "No policy information" in text

    def test_format_card_block_pending(self) -> None:
        from ui.streamlit_app import _format_card_block
        resp = {
            "action": "card_block",
            "account_number": "BSP100000001",
            "status": "pending_approval",
            "approval_request_id": "abc123def456",
            "message": "Card block request submitted.",
            "requires_approval": True,
        }
        text = _format_card_block(resp)
        assert "Card Block Request" in text
        assert "BSP100000001" in text
        assert "abc123de" in text
        assert "Awaiting human approval" in text

    def test_format_card_block_already_blocked(self) -> None:
        from ui.streamlit_app import _format_card_block
        resp = {
            "status": "already_blocked",
            "message": "This card is already blocked.",
        }
        text = _format_card_block(resp)
        assert "already blocked" in text

    def test_format_response_routing(self) -> None:
        from ui.streamlit_app import _format_response
        text = _format_response("banking", {"accounts": [], "transactions": []}, "trace123")
        assert "Account Summary" in text

        text2 = _format_response("card_block", {
            "status": "pending_approval",
            "account_number": "BSP001",
            "approval_request_id": "test123",
            "message": "Submitted.",
            "requires_approval": True,
        }, "trace456")
        assert "Card Block" in text2

        text3 = _format_response("unknown_intent", {}, "trace789")
        assert "trace78" in text3

    def test_format_response_compliance_routing(self) -> None:
        from ui.streamlit_app import _format_response
        text = _format_response("compliance", {
            "answer": "Policy says X.",
            "sources": [{"source": "test.md", "category": "Test", "heading": "Sec1"}],
        }, "trace999")
        assert "Policy says X" in text
        assert "test.md" in text


class TestStreamlitImport:
    """Verify the Streamlit app module can be imported."""

    def test_module_importable(self) -> None:
        import ui.streamlit_app
        assert hasattr(ui.streamlit_app, "_format_response")
        assert hasattr(ui.streamlit_app, "_render_trace")
        assert hasattr(ui.streamlit_app, "_init_backend")
        assert hasattr(ui.streamlit_app, "_format_card_block")
        assert hasattr(ui.streamlit_app, "_format_compliance")
