"""Tests for gap-fill implementations: expiration, PII integration,
input guard hardening, credit formatting, compliance retriever init."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from gateway.approval import (
    APPROVAL_EXPIRY_MINUTES,
    ApprovalRequest,
    ApprovalStatus,
    _check_expiry,
    clear_approval_log,
    get_pending_approvals,
    request_approval,
    resolve_approval,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_approval_log()
    yield
    clear_approval_log()


# ---------------------------------------------------------------------------
# Approval Expiration
# ---------------------------------------------------------------------------


class TestApprovalExpiration:

    def test_new_request_has_expires_at(self):
        req = request_approval(
            action="card_block",
            customer_uid="CUST001",
            agent="banking",
        )
        assert req.expires_at is not None
        expires = datetime.fromisoformat(req.expires_at)
        created = datetime.fromisoformat(req.created_at)
        diff = (expires - created).total_seconds()
        assert abs(diff - APPROVAL_EXPIRY_MINUTES * 60) < 5

    def test_expired_request_auto_marked(self):
        req = request_approval(
            action="card_block",
            customer_uid="CUST001",
            agent="banking",
        )
        req.expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()

        assert _check_expiry(req) is True
        assert req.status == ApprovalStatus.EXPIRED
        assert req.resolved_by == "system_expiry"

    def test_non_expired_request_stays_pending(self):
        req = request_approval(
            action="card_block",
            customer_uid="CUST001",
            agent="banking",
        )
        assert _check_expiry(req) is False
        assert req.status == ApprovalStatus.PENDING

    def test_resolve_expired_returns_none(self):
        req = request_approval(
            action="card_block",
            customer_uid="CUST001",
            agent="banking",
        )
        req.expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()

        result = resolve_approval(req.request_id, approved=True)
        assert result is None

    def test_get_pending_filters_expired(self):
        r1 = request_approval(action="card_block", customer_uid="CUST001", agent="banking")
        r2 = request_approval(action="card_block", customer_uid="CUST001", agent="banking")
        r1.expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()

        pending = get_pending_approvals(customer_uid="CUST001")
        assert len(pending) == 1
        assert pending[0].request_id == r2.request_id

    def test_execute_card_block_expires_pending(self):
        """A PENDING approval that has passed its deadline is auto-expired
        when execute_card_block checks it."""
        from tools.banking_tools import execute_card_block
        from ml.db_engine import init_db, get_session
        from ml.seed_data import seed_database

        engine = init_db("sqlite:///data/banksphere.db")
        seed_database("sqlite:///data/banksphere.db")
        session = get_session(engine)

        try:
            req = request_approval(
                action="card_block",
                customer_uid="CUST001",
                agent="banking",
                details={"account_number": "BSP100000001"},
            )
            req.expires_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()

            result = execute_card_block(session, req.request_id)
            assert result["status"] == "expired"
        finally:
            session.close()
            engine.dispose()


# ---------------------------------------------------------------------------
# Input Guard Hardening
# ---------------------------------------------------------------------------


class TestInputGuardHardening:

    def test_skip_input_guard_not_parameter(self):
        """Verify skip_input_guard is not a public parameter of run_supervisor."""
        import inspect
        from agents.supervisor import run_supervisor

        sig = inspect.signature(run_supervisor)
        assert "skip_input_guard" not in sig.parameters

    def test_testing_flag_exists_as_module_var(self):
        from agents import supervisor
        assert hasattr(supervisor, "_TESTING_SKIP_INPUT_GUARD")
        assert supervisor._TESTING_SKIP_INPUT_GUARD is False


# ---------------------------------------------------------------------------
# Credit Formatting
# ---------------------------------------------------------------------------


class TestCreditFormatting:

    def test_format_credit_handles_emi_key(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from ui.streamlit_app import _format_credit

        resp = {
            "eligible": True,
            "risk_category": "low",
            "default_probability": 0.05,
            "emi": "12345.67",
            "total_payment": "444444.12",
            "total_interest": "55555.88",
        }
        formatted = _format_credit(resp)
        assert "Monthly EMI" in formatted
        assert "12,345.67" in formatted
        assert "Total Payment" in formatted

    def test_format_credit_handles_monthly_emi_key(self):
        from ui.streamlit_app import _format_credit

        resp = {
            "eligible": False,
            "monthly_emi": 15000.50,
        }
        formatted = _format_credit(resp)
        assert "Monthly EMI" in formatted
        assert "15,000.50" in formatted


# ---------------------------------------------------------------------------
# PII Masking Integration
# ---------------------------------------------------------------------------


class TestPIIMaskingIntegration:

    def test_mask_pii_on_formatted_response(self):
        from gateway.pii import mask_pii

        response_text = "Contact synth_user_0001@banksphere.test for details"
        masked = mask_pii(response_text)
        assert "synth_user_0001@banksphere.test" not in masked
        assert "***@***.***" in masked


# ---------------------------------------------------------------------------
# Compliance Retriever Initialization
# ---------------------------------------------------------------------------


class TestComplianceRetrieverInit:

    def test_init_retriever_returns_non_none(self):
        from tools.compliance_tools import init_retriever, get_retriever

        retriever = init_retriever()
        assert retriever is not None
        assert get_retriever() is not None
        assert retriever.corpus_size > 0
