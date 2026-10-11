"""End-to-end tests for the card-block approval flow.

Verifies that:
- Card block creates a PENDING approval request
- The protected action cannot execute before approval
- Approval permits execution
- Rejection prevents execution
- Duplicate resolution is rejected
- Cross-customer access is denied
- Expired requests cannot be approved
- The flow works through the supervisor
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker

from gateway.approval import (
    ApprovalStatus,
    clear_approval_log,
    get_approval_log,
    get_pending_approvals,
    resolve_approval,
)
from ml.db_engine import init_db, get_session
from ml.seed_data import seed_database
from tools.banking_tools import (
    AuthorizationDenied,
    CustomerNotFound,
    execute_card_block,
    request_card_block,
)
from tools.schemas import CardBlockRequest
from tools.registry import (
    _TOOL_REGISTRY,
    register_banking_tools,
    register_fraud_tools,
    register_credit_tools,
    register_compliance_tools,
)


@pytest.fixture(scope="module")
def db_engine():
    engine = init_db("sqlite:///data/banksphere.db")
    seed_database("sqlite:///data/banksphere.db")
    return engine


@pytest.fixture(scope="module")
def _register_tools():
    _TOOL_REGISTRY.clear()
    register_banking_tools()
    register_fraud_tools()
    register_credit_tools()
    register_compliance_tools()


@pytest.fixture(scope="module")
def fraud_artifacts():
    from ml.fraud_model import train_fraud_models
    return train_fraud_models()


@pytest.fixture(scope="module")
def credit_artifacts():
    from ml.credit_model import train_credit_model
    return train_credit_model()


@pytest.fixture()
def session(db_engine):
    from sqlalchemy import select
    from ml.db_models import Account

    s = get_session(db_engine)
    yield s
    # Restore any accounts that were frozen during testing
    frozen = s.execute(
        select(Account).where(Account.status == "frozen")
    ).scalars().all()
    for acct in frozen:
        acct.status = "active"
    s.commit()
    s.close()


@pytest.fixture(autouse=True)
def _clear_approvals():
    clear_approval_log()
    yield
    clear_approval_log()


CUSTOMER_UID = "0c0e00da-99c9-c0f5-5943-5b070b004446"
ACCOUNT_NUMBER = "BSP100000001"


class TestCardBlockApprovalFlow:
    """Test the full card-block → approve → execute lifecycle."""

    def test_request_creates_pending_approval(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)
        assert resp.status == "pending_approval"
        assert resp.requires_approval is True
        assert resp.approval_request_id != ""

        pending = get_pending_approvals(customer_uid=CUSTOMER_UID)
        assert len(pending) == 1
        assert pending[0].action == "card_block"
        assert pending[0].status == ApprovalStatus.PENDING

    def test_execute_blocked_before_approval(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)
        result = execute_card_block(session, resp.approval_request_id)
        assert result["status"] == "pending"
        assert "awaiting approval" in result["message"].lower()

    def test_approve_then_execute(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)

        resolved = resolve_approval(resp.approval_request_id, approved=True, resolved_by="test")
        assert resolved is not None
        assert resolved.status == ApprovalStatus.APPROVED

        result = execute_card_block(session, resp.approval_request_id)
        assert result["status"] == "blocked"
        assert ACCOUNT_NUMBER in result["message"]

        session.rollback()

    def test_reject_prevents_execution(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)

        resolve_approval(resp.approval_request_id, approved=False, resolved_by="test")
        result = execute_card_block(session, resp.approval_request_id)
        assert result["status"] == "rejected"

    def test_duplicate_resolution_rejected(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)

        resolve_approval(resp.approval_request_id, approved=True, resolved_by="test")
        second = resolve_approval(resp.approval_request_id, approved=True, resolved_by="test")
        assert second is None

    def test_unknown_approval_id(self, session, _register_tools):
        result = execute_card_block(session, "nonexistent_id")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_cross_customer_denied(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid="OTHER_CUSTOMER_UID",
            account_number=ACCOUNT_NUMBER,
        )
        with pytest.raises(AuthorizationDenied):
            request_card_block(session, req)

    def test_nonexistent_customer(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid="nonexistent",
            requesting_customer_uid="nonexistent",
            account_number="BSP000000000",
        )
        with pytest.raises(CustomerNotFound):
            request_card_block(session, req)

    def test_nonexistent_account(self, session, _register_tools):
        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number="NONEXISTENT",
        )
        with pytest.raises(CustomerNotFound):
            request_card_block(session, req)


class TestExpiredApproval:

    def test_expired_request_cannot_execute(self, session, _register_tools):
        from gateway.approval import _approval_log

        req = CardBlockRequest(
            customer_uid=CUSTOMER_UID,
            requesting_customer_uid=CUSTOMER_UID,
            account_number=ACCOUNT_NUMBER,
        )
        resp = request_card_block(session, req)

        # Directly expire the approval in the internal log
        for r in _approval_log:
            if r.request_id == resp.approval_request_id:
                r.status = ApprovalStatus.EXPIRED
                break

        result = execute_card_block(session, resp.approval_request_id)
        assert result["status"] == "expired"


class TestCardBlockViaAgent:

    def test_banking_agent_card_block(self, session, _register_tools):
        from agents.banking_agent import BankingAgent

        agent = BankingAgent(trace_id="test-trace")
        result = agent.handle_card_block(session, CUSTOMER_UID)
        assert result.status == "completed"
        assert result.response["action"] == "card_block"
        assert result.response["requires_approval"] is True
        assert result.response["approval_request_id"] != ""

        pending = get_pending_approvals(customer_uid=CUSTOMER_UID)
        assert len(pending) >= 1


class TestCardBlockViaSupervisor:

    def test_supervisor_routes_card_block(
        self, session, _register_tools, fraud_artifacts, credit_artifacts,
    ):
        from agents.supervisor import RoutingRequest, run_supervisor

        result = run_supervisor(
            RoutingRequest(query="Block my card", customer_uid=CUSTOMER_UID),
            session=session,
            fraud_artifacts=fraud_artifacts,
            credit_artifacts=credit_artifacts,
        )
        assert result.intent == "card_block"
        assert result.status == "completed"
        resp = result.result.get("response", {})
        assert resp.get("action") == "card_block"
        assert resp.get("requires_approval") is True

        pending = get_pending_approvals(customer_uid=CUSTOMER_UID)
        assert len(pending) >= 1

    def test_supervisor_card_block_full_lifecycle(
        self, session, _register_tools, fraud_artifacts, credit_artifacts,
    ):
        from agents.supervisor import RoutingRequest, run_supervisor

        result = run_supervisor(
            RoutingRequest(query="I want to freeze my card", customer_uid=CUSTOMER_UID),
            session=session,
            fraud_artifacts=fraud_artifacts,
            credit_artifacts=credit_artifacts,
        )
        assert result.intent == "card_block"

        resp = result.result.get("response", {})
        approval_id = resp.get("approval_request_id", "")
        assert approval_id != ""

        exec_before = execute_card_block(session, approval_id)
        assert exec_before["status"] == "pending"

        resolve_approval(approval_id, approved=True, resolved_by="test_operator")

        exec_after = execute_card_block(session, approval_id)
        assert exec_after["status"] == "blocked"

        session.rollback()


class TestIntentClassification:

    @pytest.mark.parametrize("query,expected", [
        ("Block my card", "card_block"),
        ("freeze my debit card", "card_block"),
        ("I want to cancel my card", "card_block"),
        ("disable my ATM card", "card_block"),
        ("stop my credit card", "card_block"),
        ("show my balance", "banking"),
        ("card status", "banking"),
        ("What is the KYC policy?", "compliance"),
        ("Am I eligible for a loan?", "credit"),
        ("suspicious transaction", "fraud"),
    ])
    def test_classify_intent(self, query, expected):
        from agents.supervisor import classify_intent
        assert classify_intent(query) == expected
