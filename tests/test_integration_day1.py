"""Day-1 end-to-end integration tests.

Exercises the full stack: seed data → register tools → train models →
run supervisor routing for banking, fraud, credit, and unsupported queries.
Also tests direct agent calls, cross-customer denial, and edge cases.
"""

from __future__ import annotations

import warnings
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from agents.banking_agent import BankingAgent
from agents.credit_agent import CreditAgent
from agents.fraud_agent import FraudAgent
from agents.supervisor import RoutingRequest, RoutingResult, run_supervisor
from ml.credit_model import CreditModelArtifacts, train_credit_model
from ml.db_models import Account, Base, Customer, Loan, Transaction
from ml.fraud_model import FraudModelArtifacts, TransactionScoreRequest, train_fraud_models
from ml.seed_data import (
    SEED,
    _seeded_faker,
    _seeded_rng,
    generate_accounts,
    generate_customers,
    generate_loans,
    generate_transactions,
)
from tools.banking_tools import AuthorizationDenied
from tools.registry import (
    ToolPermissionDenied,
    _TOOL_REGISTRY,
    invoke_tool,
    register_banking_tools,
    register_credit_tools,
    register_fraud_tools,
)
from tools.schemas import CustomerProfileRequest, TransactionsRequest

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — set up once, shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def integration_env():
    """Full integration environment: DB, tools, models."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    fake = _seeded_faker(SEED)
    rng = _seeded_rng(SEED)
    factory = sessionmaker(bind=engine)
    session = factory()

    customers = generate_customers(session, fake, rng, n=30)
    accounts = generate_accounts(session, rng, customers)
    generate_transactions(session, fake, rng, accounts, target=1000)
    generate_loans(session, rng, customers)
    session.commit()

    _TOOL_REGISTRY.clear()
    register_banking_tools()
    register_fraud_tools()
    register_credit_tools()

    fraud_artifacts = train_fraud_models()
    credit_artifacts = train_credit_model()

    cust_a = customers[0]
    cust_b = customers[1]

    yield {
        "session": session,
        "engine": engine,
        "fraud_artifacts": fraud_artifacts,
        "credit_artifacts": credit_artifacts,
        "cust_a": cust_a,
        "cust_b": cust_b,
    }

    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# 1. Data integrity checks
# ---------------------------------------------------------------------------

class TestDataIntegrity:

    def test_customers_seeded(self, integration_env):
        session = integration_env["session"]
        count = session.execute(select(func.count(Customer.id))).scalar()
        assert count == 30

    def test_accounts_exist(self, integration_env):
        session = integration_env["session"]
        count = session.execute(select(func.count(Account.id))).scalar()
        assert count > 30

    def test_transactions_exist(self, integration_env):
        session = integration_env["session"]
        count = session.execute(select(func.count(Transaction.id))).scalar()
        assert count >= 500

    def test_loans_exist(self, integration_env):
        session = integration_env["session"]
        count = session.execute(select(func.count(Loan.id))).scalar()
        assert count > 0

    def test_foreign_keys_valid(self, integration_env):
        session = integration_env["session"]
        orphan_accounts = session.execute(
            select(Account).where(
                ~Account.customer_id.in_(select(Customer.id))
            )
        ).scalars().all()
        assert len(orphan_accounts) == 0


# ---------------------------------------------------------------------------
# 2. Balance + last 5 transactions (Scenario 1)
# ---------------------------------------------------------------------------

class TestBalanceAndTransactions:

    def test_via_banking_agent(self, integration_env):
        env = integration_env
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(
            env["session"], env["cust_a"].customer_uid,
        )
        assert result.status == "completed"
        assert len(result.response["accounts"]) >= 1
        assert result.response["total_transactions_returned"] <= 5
        for acct in result.response["accounts"]:
            assert "balance" in acct
            assert "account_number" in acct

    def test_via_supervisor(self, integration_env):
        env = integration_env
        req = RoutingRequest(
            query="What is my account balance and recent transactions?",
            customer_uid=env["cust_a"].customer_uid,
        )
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.intent == "banking"
        assert result.status == "completed"
        assert len(result.trace_id) > 0


# ---------------------------------------------------------------------------
# 3. Suspicious transaction scoring (Scenario 2)
# ---------------------------------------------------------------------------

class TestSuspiciousTransactionScoring:

    def test_high_risk_via_fraud_agent(self, integration_env):
        env = integration_env
        suspicious = TransactionScoreRequest(
            amount=Decimal("95000"),
            hour=3,
            channel="card",
            device_id="UNKNOWN-DEV",
            location_city="Chennai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-A"],
            recent_txn_count_1h=10,
            customer_avg_amount=2000,
            customer_std_amount=1500,
        )
        agent = FraudAgent()
        result = agent.score_single_transaction(suspicious, env["fraud_artifacts"])
        assert result.status == "completed"
        assert result.response["recommendation"] in ("verify", "block")
        assert result.response["fraud_score"] >= 0.3
        assert len(result.response["top_features"]) > 0
        assert result.response["is_synthetic"] is True

    def test_low_risk_via_fraud_agent(self, integration_env):
        env = integration_env
        normal = TransactionScoreRequest(
            amount=Decimal("300"),
            hour=11,
            channel="upi",
            device_id="DEV-A",
            location_city="Mumbai",
            customer_home_city="Mumbai",
            customer_known_devices=["DEV-A"],
            recent_txn_count_1h=1,
            customer_avg_amount=2000,
            customer_std_amount=1500,
        )
        agent = FraudAgent()
        result = agent.score_single_transaction(normal, env["fraud_artifacts"])
        assert result.status == "completed"
        assert result.response["recommendation"] == "allow"
        assert result.response["requires_approval"] is False

    def test_batch_scoring_via_supervisor(self, integration_env):
        env = integration_env
        req = RoutingRequest(
            query="Check for suspicious transactions on my account",
            customer_uid=env["cust_a"].customer_uid,
        )
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.intent == "fraud"
        assert result.status == "completed"

    def test_block_requires_approval(self, integration_env):
        env = integration_env
        extreme = TransactionScoreRequest(
            amount=Decimal("500000"),
            hour=2,
            channel="card",
            device_id="BRAND-NEW-DEVICE",
            location_city="Lagos",
            customer_home_city="Delhi",
            customer_known_devices=[],
            recent_txn_count_1h=15,
            customer_avg_amount=1000,
            customer_std_amount=500,
        )
        agent = FraudAgent()
        result = agent.score_single_transaction(extreme, env["fraud_artifacts"])
        if result.response["recommendation"] == "block":
            assert result.response["requires_approval"] is True


# ---------------------------------------------------------------------------
# 4. Credit risk / eligibility / EMI (Scenario 3)
# ---------------------------------------------------------------------------

class TestCreditRiskEligibilityEMI:

    def test_eligible_via_credit_agent(self, integration_env):
        env = integration_env
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=env["session"],
            customer_uid=env["cust_a"].customer_uid,
            credit_artifacts=env["credit_artifacts"],
            loan_amount=Decimal("500000"),
            credit_score=750,
            annual_income=Decimal("1500000"),
            employment_years=8,
            age=35,
        )
        assert result.status == "completed"
        assert result.response["eligible"] is True
        assert Decimal(result.response["emi"]) > 0
        assert result.response["is_synthetic"] is True

    def test_ineligible_via_credit_agent(self, integration_env):
        env = integration_env
        agent = CreditAgent()
        result = agent.assess_loan_eligibility(
            session=env["session"],
            customer_uid=env["cust_a"].customer_uid,
            credit_artifacts=env["credit_artifacts"],
            loan_amount=Decimal("5000000"),
            credit_score=400,
            annual_income=Decimal("200000"),
            employment_years=0,
            age=22,
        )
        assert result.status == "completed"
        assert result.response["eligible"] is False
        assert len(result.response["reason_codes"]) > 0

    def test_emi_calculation(self, integration_env):
        agent = CreditAgent()
        result = agent.calculate_emi_only(
            principal=Decimal("1000000"),
            annual_rate_pct=Decimal("10.5"),
            tenure_months=60,
        )
        assert result.status == "completed"
        emi = Decimal(result.response["emi"])
        assert emi > 0
        total = Decimal(result.response["total_payment"])
        assert total > Decimal("1000000")

    def test_via_supervisor(self, integration_env):
        env = integration_env
        req = RoutingRequest(
            query="Am I eligible for a personal loan?",
            customer_uid=env["cust_a"].customer_uid,
            loan_amount="500000",
            credit_score=750,
            annual_income="1500000",
            employment_years=8,
            age=35,
        )
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.intent == "credit"
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# 5. Combined routing — all three specialists
# ---------------------------------------------------------------------------

class TestCombinedRouting:

    @pytest.mark.parametrize("query,expected_intent", [
        ("Show my account balance", "banking"),
        ("What are my recent transactions?", "banking"),
        ("Is this transaction fraudulent?", "fraud"),
        ("Check for suspicious activity", "fraud"),
        ("Calculate my loan EMI", "credit"),
        ("Am I eligible to borrow?", "credit"),
        ("What is the weather?", "unsupported"),
        ("Tell me a joke", "unsupported"),
    ])
    def test_intent_routing(self, integration_env, query, expected_intent):
        env = integration_env
        req = RoutingRequest(query=query, customer_uid=env["cust_a"].customer_uid)
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.intent == expected_intent

    def test_unsupported_returns_error(self, integration_env):
        env = integration_env
        req = RoutingRequest(query="What is the meaning of life?", customer_uid=env["cust_a"].customer_uid)
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.status == "unsupported"
        assert result.error is not None


# ---------------------------------------------------------------------------
# 6. Authorization / security regression
# ---------------------------------------------------------------------------

class TestAuthorizationRegression:

    def test_cross_customer_profile_denied(self, integration_env):
        env = integration_env
        with pytest.raises(AuthorizationDenied):
            invoke_tool(
                "get_customer_profile",
                agent="banking",
                session=env["session"],
                request=CustomerProfileRequest(
                    customer_uid=env["cust_a"].customer_uid,
                    requesting_customer_uid=env["cust_b"].customer_uid,
                ),
            )

    def test_cross_customer_transactions_denied(self, integration_env):
        env = integration_env
        with pytest.raises(AuthorizationDenied):
            invoke_tool(
                "get_transactions",
                agent="banking",
                session=env["session"],
                request=TransactionsRequest(
                    customer_uid=env["cust_a"].customer_uid,
                    requesting_customer_uid=env["cust_b"].customer_uid,
                ),
            )

    def test_fraud_agent_cannot_call_profile(self, integration_env):
        env = integration_env
        with pytest.raises(ToolPermissionDenied):
            invoke_tool(
                "get_customer_profile",
                agent="fraud",
                session=env["session"],
                request=CustomerProfileRequest(
                    customer_uid=env["cust_a"].customer_uid,
                    requesting_customer_uid=env["cust_a"].customer_uid,
                ),
            )

    def test_supervisor_has_no_tool_access(self, integration_env):
        env = integration_env
        with pytest.raises(ToolPermissionDenied):
            invoke_tool(
                "get_accounts",
                agent="supervisor",
                session=env["session"],
            )


# ---------------------------------------------------------------------------
# 7. Error handling regression
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_banking_agent_missing_customer(self, integration_env):
        env = integration_env
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(env["session"], "GHOST-UID")
        assert result.status == "error"
        assert len(result.steps) > 0
        assert result.steps[-1].outcome == "error"

    def test_fraud_agent_missing_customer(self, integration_env):
        env = integration_env
        agent = FraudAgent()
        result = agent.score_customer_transactions(
            env["session"], "GHOST-UID", env["fraud_artifacts"],
        )
        assert result.status == "error"

    def test_credit_agent_untrained_model(self, integration_env):
        env = integration_env
        agent = CreditAgent()
        empty = CreditModelArtifacts()
        result = agent.assess_loan_eligibility(
            session=env["session"],
            customer_uid=env["cust_a"].customer_uid,
            credit_artifacts=empty,
            loan_amount=Decimal("100000"),
        )
        assert result.status == "error"
        assert "not trained" in result.error

    def test_supervisor_nonexistent_customer(self, integration_env):
        env = integration_env
        req = RoutingRequest(query="Show my balance", customer_uid="NOPE-999")
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert result.status == "error"


# ---------------------------------------------------------------------------
# 8. Trace / audit fields present
# ---------------------------------------------------------------------------

class TestTraceability:

    def test_banking_result_has_trace_id(self, integration_env):
        env = integration_env
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(
            env["session"], env["cust_a"].customer_uid,
        )
        assert len(result.trace_id) > 0
        assert result.agent == "banking"

    def test_fraud_result_has_trace_id(self, integration_env):
        env = integration_env
        agent = FraudAgent()
        req = TransactionScoreRequest(amount=Decimal("500"), hour=12, channel="upi")
        result = agent.score_single_transaction(req, env["fraud_artifacts"])
        assert len(result.trace_id) > 0
        assert result.agent == "fraud"

    def test_supervisor_result_has_trace_id(self, integration_env):
        env = integration_env
        req = RoutingRequest(query="Show balance", customer_uid=env["cust_a"].customer_uid)
        result = run_supervisor(
            req, env["session"], env["fraud_artifacts"], env["credit_artifacts"],
        )
        assert len(result.trace_id) == 12

    def test_steps_have_timing(self, integration_env):
        env = integration_env
        agent = BankingAgent()
        result = agent.get_balance_and_transactions(
            env["session"], env["cust_a"].customer_uid,
        )
        for step in result.steps:
            assert step.duration_ms >= 0
