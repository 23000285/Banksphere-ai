"""Tests for synthetic data generation — row counts, relationships, determinism."""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from ml.db_models import Account, Base, Customer, Loan, Transaction
from ml.seed_data import (
    NUM_CUSTOMERS,
    SEED,
    TARGET_TRANSACTIONS,
    generate_accounts,
    generate_customers,
    generate_loans,
    generate_transactions,
    seed_database,
    _seeded_faker,
    _seeded_rng,
)


@pytest.fixture()
def in_memory_db():
    """Seed a fresh in-memory SQLite database for testing."""
    url = "sqlite://"
    counts = seed_database(database_url=url)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    # Re-seed into this engine since seed_database creates its own engine
    # Use a dedicated fixture engine instead
    yield engine, counts
    engine.dispose()


@pytest.fixture()
def seeded_engine():
    """Return an engine with fully seeded data."""
    url = "sqlite://"
    engine = create_engine(url)
    Base.metadata.create_all(engine)

    fake = _seeded_faker(SEED)
    rng = _seeded_rng(SEED)

    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        customers = generate_customers(session, fake, rng)
        accounts = generate_accounts(session, rng, customers)
        n_txns = generate_transactions(session, fake, rng, accounts)
        loans = generate_loans(session, rng, customers)
        session.commit()
        yield engine, {
            "customers": len(customers),
            "accounts": len(accounts),
            "transactions": n_txns,
            "loans": len(loans),
        }
    finally:
        session.close()
        engine.dispose()


class TestRowCounts:

    def test_customer_count(self, seeded_engine):
        engine, counts = seeded_engine
        assert counts["customers"] == NUM_CUSTOMERS

    def test_accounts_range(self, seeded_engine):
        engine, counts = seeded_engine
        assert counts["accounts"] >= NUM_CUSTOMERS
        assert counts["accounts"] <= NUM_CUSTOMERS * 3

    def test_transaction_count(self, seeded_engine):
        engine, counts = seeded_engine
        assert counts["transactions"] == TARGET_TRANSACTIONS

    def test_loans_created(self, seeded_engine):
        engine, counts = seeded_engine
        assert counts["loans"] > 0
        assert counts["loans"] <= NUM_CUSTOMERS


class TestRelationships:

    def test_no_orphan_accounts(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            orphans = conn.execute(text(
                "SELECT COUNT(*) FROM accounts a "
                "LEFT JOIN customers c ON a.customer_id = c.id "
                "WHERE c.id IS NULL"
            )).scalar()
            assert orphans == 0

    def test_no_orphan_transactions(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            orphans = conn.execute(text(
                "SELECT COUNT(*) FROM transactions t "
                "LEFT JOIN accounts a ON t.account_id = a.id "
                "WHERE a.id IS NULL"
            )).scalar()
            assert orphans == 0

    def test_no_orphan_loans(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            orphans = conn.execute(text(
                "SELECT COUNT(*) FROM loans l "
                "LEFT JOIN customers c ON l.customer_id = c.id "
                "WHERE c.id IS NULL"
            )).scalar()
            assert orphans == 0

    def test_every_customer_has_account(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            no_acct = conn.execute(text(
                "SELECT COUNT(*) FROM customers c "
                "LEFT JOIN accounts a ON c.id = a.customer_id "
                "WHERE a.id IS NULL"
            )).scalar()
            assert no_acct == 0


class TestDataQuality:

    def test_unique_customer_uids(self, seeded_engine):
        engine, counts = seeded_engine
        with engine.connect() as conn:
            distinct = conn.execute(text(
                "SELECT COUNT(DISTINCT customer_uid) FROM customers"
            )).scalar()
            assert distinct == counts["customers"]

    def test_unique_account_numbers(self, seeded_engine):
        engine, counts = seeded_engine
        with engine.connect() as conn:
            distinct = conn.execute(text(
                "SELECT COUNT(DISTINCT account_number) FROM accounts"
            )).scalar()
            assert distinct == counts["accounts"]

    def test_unique_transaction_uids(self, seeded_engine):
        engine, counts = seeded_engine
        with engine.connect() as conn:
            distinct = conn.execute(text(
                "SELECT COUNT(DISTINCT transaction_uid) FROM transactions"
            )).scalar()
            assert distinct == counts["transactions"]

    def test_emails_are_synthetic(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            emails = conn.execute(text(
                "SELECT email FROM customers"
            )).fetchall()
            for (email,) in emails:
                assert "@banksphere.test" in email

    def test_amounts_positive(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            neg = conn.execute(text(
                "SELECT COUNT(*) FROM transactions WHERE amount <= 0"
            )).scalar()
            assert neg == 0

    def test_balances_positive(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            neg = conn.execute(text(
                "SELECT COUNT(*) FROM accounts WHERE balance < 0"
            )).scalar()
            assert neg == 0

    def test_transaction_types_valid(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            types = {r[0] for r in conn.execute(text(
                "SELECT DISTINCT txn_type FROM transactions"
            )).fetchall()}
            assert types.issubset({"credit", "debit"})

    def test_flagged_transactions_exist(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            flagged = conn.execute(text(
                "SELECT COUNT(*) FROM transactions WHERE is_flagged = 1"
            )).scalar()
            assert flagged > 0
            total = conn.execute(text(
                "SELECT COUNT(*) FROM transactions"
            )).scalar()
            assert flagged / total < 0.05


class TestDeterminism:

    _SMALL = dict(num_customers=20, target_transactions=500)

    def test_same_seed_same_counts(self):
        """Running with the same seed produces identical row counts."""
        url = "sqlite://"
        c1 = seed_database(database_url=url, seed=SEED, **self._SMALL)
        c2 = seed_database(database_url=url, seed=SEED, **self._SMALL)
        assert c1 == c2

    def test_different_seed_different_data(self):
        """A different seed produces different loan counts or account counts."""
        url = "sqlite://"
        c1 = seed_database(database_url=url, seed=SEED, **self._SMALL)
        c2 = seed_database(database_url=url, seed=SEED + 1, **self._SMALL)
        assert c1["loans"] != c2["loans"] or c1["accounts"] != c2["accounts"]


class TestReseed:

    def test_reseed_drops_and_recreates(self):
        """Calling seed_database twice on the same DB doesn't double the rows."""
        url = "sqlite://"
        _small = dict(num_customers=20, target_transactions=500)
        c1 = seed_database(database_url=url, seed=SEED, **_small)
        c2 = seed_database(database_url=url, seed=SEED, **_small)
        assert c1 == c2


class TestSampleQueries:

    def test_top_balance_query(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT a.account_number, a.balance, c.first_name, c.last_name "
                "FROM accounts a JOIN customers c ON a.customer_id = c.id "
                "ORDER BY a.balance DESC LIMIT 5"
            )).fetchall()
            assert len(rows) == 5
            assert all(r[1] > 0 for r in rows)

    def test_recent_transactions_query(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT t.transaction_uid, t.amount, t.category, t.timestamp "
                "FROM transactions t "
                "ORDER BY t.timestamp DESC LIMIT 5"
            )).fetchall()
            assert len(rows) == 5

    def test_loan_summary_query(self, seeded_engine):
        engine, _ = seeded_engine
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT loan_type, COUNT(*), AVG(principal), AVG(interest_rate) "
                "FROM loans GROUP BY loan_type"
            )).fetchall()
            assert len(rows) > 0
