"""Tests for banking tools — authorization, queries, edge cases, and registry."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ml.db_models import Base
from ml.seed_data import (
    SEED,
    _seeded_faker,
    _seeded_rng,
    generate_accounts,
    generate_customers,
    generate_transactions,
    generate_loans,
)
from tools.banking_tools import (
    AuthorizationDenied,
    CustomerNotFound,
    get_accounts,
    get_card_status,
    get_customer_profile,
    get_transactions,
)
from tools.schemas import (
    AccountsRequest,
    CardStatusRequest,
    CustomerProfileRequest,
    TransactionsRequest,
)
from tools.registry import (
    ToolPermissionDenied,
    agent_allowed_tools,
    invoke_tool,
    register_banking_tools,
    list_tools,
)


@pytest.fixture(scope="module")
def seeded_session():
    """Create an in-memory DB with a small dataset for tool testing."""
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

    yield session, customers, accounts

    session.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _register_tools():
    """Ensure banking tools are registered before each test."""
    register_banking_tools()


# ---------------------------------------------------------------------------
# Customer profile
# ---------------------------------------------------------------------------

class TestGetCustomerProfile:

    def test_valid_request(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = CustomerProfileRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
        )
        resp = get_customer_profile(session, req)
        assert resp.customer_uid == cust.customer_uid
        assert resp.first_name == cust.first_name
        assert resp.account_count >= 1

    def test_cross_customer_denied(self, seeded_session):
        session, customers, _ = seeded_session
        req = CustomerProfileRequest(
            customer_uid=customers[0].customer_uid,
            requesting_customer_uid=customers[1].customer_uid,
        )
        with pytest.raises(AuthorizationDenied):
            get_customer_profile(session, req)

    def test_missing_customer(self, seeded_session):
        session, _, _ = seeded_session
        req = CustomerProfileRequest(
            customer_uid="nonexistent-uid-00000",
            requesting_customer_uid="nonexistent-uid-00000",
        )
        with pytest.raises(CustomerNotFound):
            get_customer_profile(session, req)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

class TestGetAccounts:

    def test_valid_request(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = AccountsRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
        )
        resp = get_accounts(session, req)
        assert resp.customer_uid == cust.customer_uid
        assert len(resp.accounts) >= 1
        assert all(a.balance >= 0 for a in resp.accounts)

    def test_cross_customer_denied(self, seeded_session):
        session, customers, _ = seeded_session
        req = AccountsRequest(
            customer_uid=customers[0].customer_uid,
            requesting_customer_uid=customers[2].customer_uid,
        )
        with pytest.raises(AuthorizationDenied):
            get_accounts(session, req)

    def test_missing_customer(self, seeded_session):
        session, _, _ = seeded_session
        req = AccountsRequest(
            customer_uid="nonexistent-uid-00000",
            requesting_customer_uid="nonexistent-uid-00000",
        )
        with pytest.raises(CustomerNotFound):
            get_accounts(session, req)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

class TestGetTransactions:

    def test_valid_default_limit(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = TransactionsRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
        )
        resp = get_transactions(session, req)
        assert resp.customer_uid == cust.customer_uid
        assert resp.total_returned <= 5
        assert len(resp.transactions) == resp.total_returned

    def test_custom_limit(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = TransactionsRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
            limit=10,
        )
        resp = get_transactions(session, req)
        assert resp.total_returned <= 10

    def test_transactions_sorted_descending(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = TransactionsRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
            limit=20,
        )
        resp = get_transactions(session, req)
        if len(resp.transactions) > 1:
            timestamps = [t.timestamp for t in resp.transactions]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_cross_customer_denied(self, seeded_session):
        session, customers, _ = seeded_session
        req = TransactionsRequest(
            customer_uid=customers[0].customer_uid,
            requesting_customer_uid=customers[3].customer_uid,
        )
        with pytest.raises(AuthorizationDenied):
            get_transactions(session, req)

    def test_missing_customer(self, seeded_session):
        session, _, _ = seeded_session
        req = TransactionsRequest(
            customer_uid="nonexistent-uid-00000",
            requesting_customer_uid="nonexistent-uid-00000",
        )
        with pytest.raises(CustomerNotFound):
            get_transactions(session, req)

    def test_invalid_limit_too_low(self):
        with pytest.raises(Exception):
            TransactionsRequest(
                customer_uid="x", requesting_customer_uid="x", limit=0,
            )

    def test_invalid_limit_too_high(self):
        with pytest.raises(Exception):
            TransactionsRequest(
                customer_uid="x", requesting_customer_uid="x", limit=200,
            )


# ---------------------------------------------------------------------------
# Card status
# ---------------------------------------------------------------------------

class TestGetCardStatus:

    def test_valid_request(self, seeded_session):
        session, customers, accounts = seeded_session
        cust = customers[0]
        acct = next(a for a in accounts if a.customer_id == cust.id)
        req = CardStatusRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
            account_number=acct.account_number,
        )
        resp = get_card_status(session, req)
        assert resp.customer_uid == cust.customer_uid
        assert resp.account_number == acct.account_number
        assert resp.status in ("active", "frozen", "closed")
        assert len(resp.message) > 0

    def test_cross_customer_denied(self, seeded_session):
        session, customers, accounts = seeded_session
        acct = next(a for a in accounts if a.customer_id == customers[0].id)
        req = CardStatusRequest(
            customer_uid=customers[0].customer_uid,
            requesting_customer_uid=customers[1].customer_uid,
            account_number=acct.account_number,
        )
        with pytest.raises(AuthorizationDenied):
            get_card_status(session, req)

    def test_wrong_account_number(self, seeded_session):
        session, customers, _ = seeded_session
        cust = customers[0]
        req = CardStatusRequest(
            customer_uid=cust.customer_uid,
            requesting_customer_uid=cust.customer_uid,
            account_number="BSP000000000",
        )
        with pytest.raises(CustomerNotFound):
            get_card_status(session, req)

    def test_missing_customer(self, seeded_session):
        session, _, _ = seeded_session
        req = CardStatusRequest(
            customer_uid="nonexistent-uid-00000",
            requesting_customer_uid="nonexistent-uid-00000",
            account_number="BSP000000000",
        )
        with pytest.raises(CustomerNotFound):
            get_card_status(session, req)


# ---------------------------------------------------------------------------
# Registry and permissions
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_banking_tools_registered(self):
        tools = list_tools()
        assert "get_customer_profile" in tools
        assert "get_accounts" in tools
        assert "get_transactions" in tools
        assert "get_card_status" in tools

    def test_banking_agent_permissions(self):
        allowed = agent_allowed_tools("banking")
        assert "get_customer_profile" in allowed
        assert "get_accounts" in allowed
        assert "get_transactions" in allowed
        assert "get_card_status" in allowed

    def test_supervisor_has_no_tools(self):
        allowed = agent_allowed_tools("supervisor")
        assert len(allowed) == 0

    def test_fraud_cannot_call_profile(self):
        assert "get_customer_profile" not in agent_allowed_tools("fraud")

    def test_invoke_denied_for_wrong_agent(self, seeded_session):
        session, customers, _ = seeded_session
        with pytest.raises(ToolPermissionDenied, match="supervisor"):
            invoke_tool(
                "get_customer_profile",
                agent="supervisor",
                session=session,
                request=CustomerProfileRequest(
                    customer_uid=customers[0].customer_uid,
                    requesting_customer_uid=customers[0].customer_uid,
                ),
            )

    def test_invoke_success_for_banking_agent(self, seeded_session):
        session, customers, _ = seeded_session
        result = invoke_tool(
            "get_customer_profile",
            agent="banking",
            session=session,
            request=CustomerProfileRequest(
                customer_uid=customers[0].customer_uid,
                requesting_customer_uid=customers[0].customer_uid,
            ),
        )
        assert result.customer_uid == customers[0].customer_uid

    def test_invoke_unregistered_tool(self):
        with pytest.raises(ToolPermissionDenied):
            invoke_tool("nonexistent_tool", agent="banking")

    def test_unknown_agent_denied(self):
        with pytest.raises(ToolPermissionDenied, match="unknown_agent"):
            invoke_tool("get_customer_profile", agent="unknown_agent")


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_missing_required_field(self):
        with pytest.raises(Exception):
            CustomerProfileRequest(customer_uid="abc")

    def test_valid_construction(self):
        req = CustomerProfileRequest(
            customer_uid="abc", requesting_customer_uid="abc",
        )
        assert req.customer_uid == "abc"
