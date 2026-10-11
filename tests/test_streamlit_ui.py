"""Tests for Phase 12: Streamlit UI helpers and importability."""

from __future__ import annotations


class TestFormatHelpers:
    """Test response formatting functions without running Streamlit."""

    def _import_helpers(self):
        import importlib
        import sys
        # Import the module's functions without triggering st.set_page_config
        # by loading just the function source
        from ui.streamlit_app import (
            _format_banking,
            _format_fraud,
            _format_credit,
            _format_compliance,
            _format_response,
        )
        return _format_banking, _format_fraud, _format_credit, _format_compliance, _format_response

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
        assert "🚩" in text

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
        assert "🔴" in text  # block
        assert "🟢" in text  # allow
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

    def test_format_compliance(self) -> None:
        from ui.streamlit_app import _format_compliance
        resp = {"answer": "KYC verification requires government-issued photo ID."}
        text = _format_compliance(resp)
        assert "KYC verification" in text

    def test_format_compliance_no_evidence(self) -> None:
        from ui.streamlit_app import _format_compliance
        text = _format_compliance({})
        assert "No policy information" in text

    def test_format_response_routing(self) -> None:
        from ui.streamlit_app import _format_response
        text = _format_response("banking", {"accounts": [], "transactions": []}, "trace123")
        assert "Account Summary" in text

        text2 = _format_response("unknown_intent", {}, "trace456")
        assert "trace45" in text2


class TestStreamlitImport:
    """Verify the Streamlit app module can be imported."""

    def test_module_importable(self) -> None:
        import ui.streamlit_app
        assert hasattr(ui.streamlit_app, "_format_response")
        assert hasattr(ui.streamlit_app, "_render_trace")
        assert hasattr(ui.streamlit_app, "_init_backend")
