"""Tests for banking and fraud agents — tool calls, step limits, error handling."""

from __future__ import annotations

import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agents.banking_agent import BankingAgent
from agents.fraud_agent import FraudAgent
from ml.db_models import Base
from ml.fraud_model import (
    FraudModelArtifacts,
    TransactionScoreRequest,
    train_fraud_models,
)
from ml.seed_data import (
    SEED,
    _seeded_faker,
    _seeded_rng,
    generate_accounts,
    generate_customers,
    generate_transactions,
    generate_loans,
)
from tools.registry import (
    register_banking_tools,
    register_fraud_tools,
    _TOOL_REGISTRY,
)

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def db_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    fake = _seeded_faker(SEED)
    rng = _seeded_rng(SEED)

    factory = sessionmaker(bind=engine)
    session = factory()

    customers = generate_customers(session, fake, rng, n=20)
    accounts = generate_accounts(session, rng, customers)
    generate_transactions(session, fake, rng, accounts, target=500)
    generate_loans(session, rng, customers)
    session.commit()

    _TOOL_REGISTRY.clear()
    register_banking_tools()
    register_fraud_tools()

    yield session
    session.close()
    engine.dispose()


@pytest.fixture(scope="module")
def customer_uid(db_session):
    from ml.db_models import Customer
    from sqlalchemy import select

    row = db_session.execute(select(Customer).limit(1)).scalar_one()
    return row.customer_uid


@pytest.fixture(scope="module")
def fraud_artifacts() -> FraudModelArtifacts:
    return train_fraud_models()


# ---------------------------------------------------------------------------
# Banking agent — balance + last 5 transactions
# ---------------------------------------------------------------------------

class TestBankingAgentBalanceAndTransactions:

    def test_returns_completed(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, customer_uid)
        assert result.status == "completed"
        assert result.agent == "banking"
        assert len(result.trace_id) > 0

    def test_response_has_accounts(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, customer_uid)
        assert "accounts" in result.response
        assert len(result.response["accounts"]) > 0
        acct = result.response["accounts"][0]
        assert "balance" in acct
        assert "account_type" in acct

    def test_response_has_transactions(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, customer_uid)
        assert "transactions" in result.response
        assert result.response["total_transactions_returned"] <= 5

    def test_steps_recorded(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, customer_uid)
        assert len(result.steps) == 2
        assert result.steps[0].tool_name == "get_accounts"
        assert result.steps[1].tool_name == "get_transactions"
        for step in result.steps:
            assert step.outcome == "success"

    def test_custom_txn_limit(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, customer_uid, txn_limit=3)
        assert result.response["total_transactions_returned"] <= 3

    def test_nonexistent_customer(self, db_session):
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(db_session, "NO-SUCH-UID")
        assert result.status == "error"
        assert "not_found" in result.steps[-1].result_summary.lower() or "no customer" in result.steps[-1].result_summary.lower()


class TestBankingAgentProfile:

    def test_returns_profile(self, db_session, customer_uid):
        agent = BankingAgent()
        result = agent.get_customer_profile(db_session, customer_uid)
        assert result.status == "completed"
        assert result.response["customer_uid"] == customer_uid
        assert "first_name" in result.response

    def test_profile_nonexistent_customer(self, db_session):
        agent = BankingAgent()
        result = agent.get_customer_profile(db_session, "MISSING-UID")
        assert result.status == "error"


class TestBankingAgentStepLimit:

    def test_max_steps_enforced(self, db_session, customer_uid):
        agent = BankingAgent(max_steps=1)
        result = agent.get_balance_and_transactions(db_session, customer_uid)
        assert len(result.steps) <= 2
        if result.status == "max_steps_reached":
            assert "step limit" in result.error


# ---------------------------------------------------------------------------
# Fraud agent — suspicious transaction scoring
# ---------------------------------------------------------------------------

class TestFraudAgentSingleTransaction:

    def _suspicious_request(self) -> TransactionScoreRequest:
        return TransactionScoreRequest(
            amount=Decimal("80000"),
            hour=2,
            channel="card",
            device_id="DEV-NEW-999",
            location_city="Chennai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-1234"],
            recent_txn_count_1h=8,
            customer_avg_amount=3000,
            customer_std_amount=2000,
        )

    def _normal_request(self) -> TransactionScoreRequest:
        return TransactionScoreRequest(
            amount=Decimal("500"),
            hour=14,
            channel="upi",
            device_id="DEV-1234",
            location_city="Mumbai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-1234"],
            recent_txn_count_1h=1,
            customer_avg_amount=3000,
            customer_std_amount=2000,
        )

    def test_suspicious_returns_block_or_verify(self, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_single_transaction(self._suspicious_request(), fraud_artifacts)
        assert result.status == "completed"
        assert result.response["recommendation"] in ("verify", "block")
        assert result.response["fraud_score"] >= 0.3

    def test_suspicious_requires_approval(self, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_single_transaction(self._suspicious_request(), fraud_artifacts)
        if result.response["recommendation"] == "block":
            assert result.response["requires_approval"] is True

    def test_normal_returns_allow(self, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_single_transaction(self._normal_request(), fraud_artifacts)
        assert result.status == "completed"
        assert result.response["recommendation"] == "allow"
        assert result.response["requires_approval"] is False

    def test_response_fields(self, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_single_transaction(self._normal_request(), fraud_artifacts)
        resp = result.response
        assert "fraud_score" in resp
        assert "anomaly_score" in resp
        assert "risk_category" in resp
        assert "top_features" in resp
        assert "explanation_method" in resp
        assert resp["is_synthetic"] is True

    def test_step_recorded(self, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_single_transaction(self._normal_request(), fraud_artifacts)
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "score_transaction"
        assert result.steps[0].outcome == "success"

    def test_untrained_model_error(self):
        agent = FraudAgent()
        empty = FraudModelArtifacts()
        req = TransactionScoreRequest(amount=Decimal("100"), hour=12, channel="upi")
        result = agent.score_single_transaction(req, empty)
        assert result.status == "error"
        assert "not trained" in result.error


class TestFraudAgentBatchScoring:

    def test_score_customer_transactions(self, db_session, customer_uid, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_customer_transactions(
            db_session, customer_uid, fraud_artifacts, txn_limit=3,
        )
        assert result.status == "completed"
        assert result.response["customer_uid"] == customer_uid
        scored = result.response["scored_transactions"]
        assert len(scored) <= 3
        for s in scored:
            assert "fraud_score" in s
            assert "recommendation" in s
            assert s["recommendation"] in ("allow", "verify", "block")

    def test_each_scored_has_uid(self, db_session, customer_uid, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_customer_transactions(
            db_session, customer_uid, fraud_artifacts, txn_limit=2,
        )
        for s in result.response["scored_transactions"]:
            assert "transaction_uid" in s

    def test_nonexistent_customer_error(self, db_session, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_customer_transactions(
            db_session, "BAD-UID", fraud_artifacts,
        )
        assert result.status == "error"

    def test_steps_include_get_transactions(self, db_session, customer_uid, fraud_artifacts):
        agent = FraudAgent()
        result = agent.score_customer_transactions(
            db_session, customer_uid, fraud_artifacts, txn_limit=2,
        )
        tool_names = [s.tool_name for s in result.steps]
        assert "get_transactions" in tool_names


class TestFraudAgentStepLimit:

    def test_max_steps_enforced(self, db_session, customer_uid, fraud_artifacts):
        agent = FraudAgent(max_steps=2)
        result = agent.score_customer_transactions(
            db_session, customer_uid, fraud_artifacts, txn_limit=5,
        )
        assert len(result.steps) <= 3
