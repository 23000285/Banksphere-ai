"""Tests for credit agent, supervisor routing, and LangGraph integration."""

from __future__ import annotations

import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from agents.credit_agent import CreditAgent
from agents.supervisor import (
    RoutingRequest,
    RoutingResult,
    classify_intent,
    run_supervisor,
)
from ml.credit_model import CreditModelArtifacts, train_credit_model
from ml.db_models import Base, Customer
from ml.fraud_model import FraudModelArtifacts, train_fraud_models
from ml.seed_data import (
    SEED,
    _seeded_faker,
    _seeded_rng,
    generate_accounts,
    generate_customers,
    generate_loans,
    generate_transactions,
)
from tools.registry import (
    _TOOL_REGISTRY,
    register_banking_tools,
    register_credit_tools,
    register_fraud_tools,
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
    register_credit_tools()

    yield session
    session.close()
    engine.dispose()


@pytest.fixture(scope="module")
def customer_uid(db_session):
    row = db_session.execute(select(Customer).limit(1)).scalar_one()
    return row.customer_uid


@pytest.fixture(scope="module")
def fraud_artifacts() -> FraudModelArtifacts:
    return train_fraud_models()


@pytest.fixture(scope="module")
def credit_artifacts() -> CreditModelArtifacts:
    return train_credit_model()


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

class TestIntentClassification:

    def test_banking_balance(self):
        assert classify_intent("What is my account balance?") == "banking"

    def test_banking_transactions(self):
        assert classify_intent("Show me recent transactions") == "banking"

    def test_banking_card(self):
        assert classify_intent("What is my card status?") == "banking"

    def test_fraud_suspicious(self):
        assert classify_intent("Is this transaction suspicious?") == "fraud"

    def test_fraud_risk_score(self):
        assert classify_intent("What is the fraud risk score?") == "fraud"

    def test_credit_loan(self):
        assert classify_intent("Am I eligible for a loan?") == "credit"

    def test_credit_emi(self):
        assert classify_intent("Calculate my EMI") == "credit"

    def test_credit_interest_rate(self):
        assert classify_intent("What interest rate can I get?") == "credit"

    def test_unsupported(self):
        assert classify_intent("What is the weather today?") == "unsupported"

    def test_empty_after_validation(self):
        assert classify_intent("hello") == "unsupported"

    def test_fraud_before_banking(self):
        assert classify_intent("Flag this suspicious transaction") == "fraud"


# ---------------------------------------------------------------------------
# Credit agent
# ---------------------------------------------------------------------------

class TestCreditAgent:

    def test_eligible_applicant(self, db_session, customer_uid, credit_artifacts):
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=db_session,
            customer_uid=customer_uid,
            credit_artifacts=credit_artifacts,
            loan_amount=Decimal("500000"),
            credit_score=750,
            annual_income=Decimal("1500000"),
            employment_years=8,
            age=35,
        )
        assert result.status == "completed"
        assert result.response["eligible"] is True
        assert result.response["customer_uid"] == customer_uid

    def test_ineligible_applicant(self, db_session, customer_uid, credit_artifacts):
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=db_session,
            customer_uid=customer_uid,
            credit_artifacts=credit_artifacts,
            loan_amount=Decimal("5000000"),
            credit_score=450,
            annual_income=Decimal("200000"),
            employment_years=0,
            age=22,
        )
        assert result.status == "completed"
        assert result.response["eligible"] is False
        assert len(result.response["reason_codes"]) > 0

    def test_steps_recorded(self, db_session, customer_uid, credit_artifacts):
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=db_session,
            customer_uid=customer_uid,
            credit_artifacts=credit_artifacts,
            loan_amount=Decimal("500000"),
        )
        assert len(result.steps) == 2
        assert result.steps[0].tool_name == "get_customer_profile"
        assert result.steps[1].tool_name == "assess_credit_risk"

    def test_nonexistent_customer(self, db_session, credit_artifacts):
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=db_session,
            customer_uid="NO-SUCH-UID",
            credit_artifacts=credit_artifacts,
            loan_amount=Decimal("500000"),
        )
        assert result.status == "error"

    def test_untrained_model(self, db_session, customer_uid):
        agent = CreditAgent()
        empty = CreditModelArtifacts()
        result = agent.assess_loan_eligibility(
            session=db_session,
            customer_uid=customer_uid,
            credit_artifacts=empty,
            loan_amount=Decimal("500000"),
        )
        assert result.status == "error"
        assert "not trained" in result.error

    def test_emi_only(self):
        agent = CreditAgent()
        result = agent.calculate_emi_only(
            principal=Decimal("500000"),
            annual_rate_pct=Decimal("12"),
            tenure_months=36,
        )
        assert result.status == "completed"
        assert "emi" in result.response
        assert Decimal(result.response["emi"]) > 0


# ---------------------------------------------------------------------------
# Supervisor routing — end to end via LangGraph
# ---------------------------------------------------------------------------

class TestSupervisorRouting:

    def test_routes_banking(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="Show my account balance", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert isinstance(result, RoutingResult)
        assert result.intent == "banking"
        assert result.specialist == "banking"
        assert result.status == "completed"
        assert len(result.trace_id) > 0

    def test_routes_fraud(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="Is this transaction suspicious?", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert result.intent == "fraud"
        assert result.specialist == "fraud"
        assert result.status == "completed"

    def test_routes_credit(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(
            query="Am I eligible for a personal loan?",
            customer_uid=customer_uid,
            loan_amount="500000",
            credit_score=750,
            annual_income="1500000",
            employment_years=8,
            age=35,
        )
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert result.intent == "credit"
        assert result.specialist == "credit"
        assert result.status == "completed"

    def test_unsupported_query(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="What is the weather?", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert result.intent == "unsupported"
        assert result.status == "unsupported"
        assert result.error is not None

    def test_result_has_trace_id(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="Show my balance", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert len(result.trace_id) == 12

    def test_result_contains_agent_data(self, db_session, customer_uid, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="Check my account balance", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert "response" in result.result or "steps" in result.result


# ---------------------------------------------------------------------------
# Malformed / edge-case inputs
# ---------------------------------------------------------------------------

class TestMalformedInputs:

    def test_empty_query_rejected(self):
        with pytest.raises(Exception):
            RoutingRequest(query="", customer_uid="some-uid")

    def test_empty_customer_uid_rejected(self):
        with pytest.raises(Exception):
            RoutingRequest(query="Balance please", customer_uid="")

    def test_nonexistent_customer(self, db_session, fraud_artifacts, credit_artifacts):
        req = RoutingRequest(query="Show my balance", customer_uid="NONEXISTENT-UID-999")
        result = run_supervisor(req, db_session, fraud_artifacts, credit_artifacts)
        assert result.status == "error"

    def test_credit_without_artifacts(self, db_session, customer_uid):
        req = RoutingRequest(query="Am I eligible for a loan?", customer_uid=customer_uid)
        result = run_supervisor(req, db_session, fraud_artifacts=None, credit_artifacts=None)
        assert result.status == "error"


# ---------------------------------------------------------------------------
# Routing request Pydantic validation
# ---------------------------------------------------------------------------

class TestRoutingRequestValidation:

    def test_valid_construction(self):
        req = RoutingRequest(query="Balance check", customer_uid="C-001")
        assert req.query == "Balance check"
        assert req.loan_tenure_months == 36

    def test_defaults(self):
        req = RoutingRequest(query="test", customer_uid="C-001")
        assert req.loan_amount is None
        assert req.credit_score == 700
        assert req.age == 30
