"""Repeatable synthetic banking data generator.

Uses Faker and NumPy with controlled seeds for deterministic output.
All data is entirely artificial — no real personal or financial information.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

from ml.db_engine import get_session, init_db
from ml.db_models import Account, Base, Customer, Loan, Transaction

SEED = 42
NUM_CUSTOMERS = 200
ACCOUNTS_PER_CUSTOMER = (1, 3)
TARGET_TRANSACTIONS = 50_000
LOANS_FRACTION = 0.35

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Indore", "Bhopal", "Kochi", "Coimbatore",
]

INDIAN_STATES = [
    "Maharashtra", "Delhi", "Karnataka", "Telangana", "Tamil Nadu",
    "West Bengal", "Maharashtra", "Gujarat", "Rajasthan", "Uttar Pradesh",
    "Punjab", "Madhya Pradesh", "Madhya Pradesh", "Kerala", "Tamil Nadu",
]

MERCHANT_CATEGORIES = {
    "groceries": ["BigBasket", "DMart", "Reliance Fresh", "More Supermarket"],
    "food": ["Swiggy", "Zomato", "Dominos", "McDonalds India"],
    "fuel": ["Indian Oil", "HP Petrol", "Bharat Petroleum"],
    "utilities": ["Electricity Board", "Water Authority", "Gas Agency", "BSNL"],
    "shopping": ["Flipkart", "Amazon India", "Myntra", "Ajio"],
    "travel": ["IRCTC", "MakeMyTrip", "RedBus", "Ola"],
    "medical": ["Apollo Pharmacy", "MedPlus", "Practo", "1mg"],
    "education": ["Byju's", "Unacademy", "School Fees", "College Fees"],
    "entertainment": ["BookMyShow", "Netflix India", "Hotstar", "PVR Cinemas"],
    "transfer": ["Peer Transfer", "Family Transfer", "Rent Payment"],
}

CHANNELS = ["upi", "neft", "imps", "card", "atm", "branch"]
CHANNEL_WEIGHTS = [0.40, 0.10, 0.10, 0.25, 0.10, 0.05]


def _seeded_faker(seed: int = SEED) -> Faker:
    fake = Faker("en_IN")
    Faker.seed(seed)
    return fake


def _seeded_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_customers(
    session: Session,
    fake: Faker,
    rng: np.random.Generator,
    n: int = NUM_CUSTOMERS,
) -> list[Customer]:
    customers = []
    for i in range(n):
        city_idx = rng.integers(0, len(INDIAN_CITIES))
        dob = fake.date_of_birth(minimum_age=21, maximum_age=65)
        created = fake.date_time_between(start_date="-3y", end_date="-30d")

        kyc_roll = rng.random()
        kyc = "verified" if kyc_roll < 0.85 else ("pending" if kyc_roll < 0.95 else "rejected")

        risk_roll = rng.random()
        risk = "low" if risk_roll < 0.60 else ("medium" if risk_roll < 0.90 else "high")

        c = Customer(
            customer_uid=str(uuid.UUID(int=int(rng.integers(0, 2**63)) ^ (int(rng.integers(0, 2**63)) << 64))),
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            email=f"synth_user_{i:04d}@banksphere.test",
            phone=fake.phone_number(),
            date_of_birth=datetime(dob.year, dob.month, dob.day),
            address=fake.street_address(),
            city=INDIAN_CITIES[city_idx],
            state=INDIAN_STATES[city_idx],
            pin_code=fake.postcode(),
            kyc_status=kyc,
            risk_category=risk,
            created_at=created,
        )
        customers.append(c)

    session.add_all(customers)
    session.flush()
    return customers


def generate_accounts(
    session: Session,
    rng: np.random.Generator,
    customers: list[Customer],
) -> list[Account]:
    accounts = []
    acct_counter = 100_000_001

    types = ["savings", "current", "salary"]
    type_weights = [0.55, 0.25, 0.20]

    for cust in customers:
        n_accts = int(rng.integers(ACCOUNTS_PER_CUSTOMER[0], ACCOUNTS_PER_CUSTOMER[1] + 1))
        for _ in range(n_accts):
            balance = Decimal(str(round(float(rng.lognormal(10, 1.5)), 2)))
            balance = min(balance, Decimal("9999999.99"))
            acct_type = rng.choice(types, p=type_weights)
            opened = cust.created_at + timedelta(days=int(rng.integers(0, 60)))

            a = Account(
                account_number=f"BSP{acct_counter}",
                customer_id=cust.id,
                account_type=acct_type,
                balance=balance,
                currency="INR",
                status="active" if rng.random() < 0.95 else "frozen",
                opened_at=opened,
            )
            accounts.append(a)
            acct_counter += 1

    session.add_all(accounts)
    session.flush()
    return accounts


def generate_transactions(
    session: Session,
    fake: Faker,
    rng: np.random.Generator,
    accounts: list[Account],
    target: int = TARGET_TRANSACTIONS,
) -> int:
    categories = list(MERCHANT_CATEGORIES.keys())
    cat_weights = np.array([0.18, 0.14, 0.08, 0.10, 0.15, 0.08, 0.05, 0.04, 0.06, 0.12])
    cat_weights = cat_weights / cat_weights.sum()

    txns_per_account = rng.multinomial(target, np.ones(len(accounts)) / len(accounts))
    batch = []
    total = 0
    batch_size = 5000

    for acct, n_txns in zip(accounts, txns_per_account):
        for _ in range(n_txns):
            cat = rng.choice(categories, p=cat_weights)
            merchants = MERCHANT_CATEGORIES[cat]
            merchant = merchants[int(rng.integers(0, len(merchants)))]

            if cat in ("transfer", "utilities"):
                amount = Decimal(str(round(float(rng.lognormal(8, 1.0)), 2)))
            else:
                amount = Decimal(str(round(float(rng.lognormal(6, 1.2)), 2)))
            amount = min(amount, Decimal("999999.99"))

            txn_type = "debit" if rng.random() < 0.70 else "credit"
            channel = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)
            is_flagged = bool(rng.random() < 0.015)

            ts = fake.date_time_between(start_date=acct.opened_at, end_date="now")

            device_id = f"DEV-{rng.integers(1000, 9999)}" if channel in ("upi", "card") else None
            location = INDIAN_CITIES[int(rng.integers(0, len(INDIAN_CITIES)))]

            t = Transaction(
                transaction_uid=str(uuid.UUID(int=int(rng.integers(0, 2**63)) ^ (int(rng.integers(0, 2**63)) << 64))),
                account_id=acct.id,
                txn_type=txn_type,
                category=cat,
                amount=amount,
                merchant=merchant,
                channel=channel,
                device_id=device_id,
                location_city=location,
                is_flagged=is_flagged,
                timestamp=ts,
            )
            batch.append(t)
            total += 1

            if len(batch) >= batch_size:
                session.add_all(batch)
                session.flush()
                batch = []

    if batch:
        session.add_all(batch)
        session.flush()

    return total


def generate_loans(
    session: Session,
    rng: np.random.Generator,
    customers: list[Customer],
    fraction: float = LOANS_FRACTION,
) -> list[Loan]:
    n_borrowers = int(len(customers) * fraction)
    borrower_indices = rng.choice(len(customers), size=n_borrowers, replace=False)
    loans = []

    loan_types = ["personal", "home", "auto", "education"]
    type_weights = [0.40, 0.25, 0.20, 0.15]

    for idx in borrower_indices:
        cust = customers[idx]
        loan_type = rng.choice(loan_types, p=type_weights)

        if loan_type == "home":
            principal = Decimal(str(round(float(rng.uniform(20_00_000, 80_00_000)), 2)))
            tenure = int(rng.choice([120, 180, 240]))
            rate = Decimal(str(round(float(rng.uniform(7.5, 10.0)), 2)))
        elif loan_type == "auto":
            principal = Decimal(str(round(float(rng.uniform(3_00_000, 15_00_000)), 2)))
            tenure = int(rng.choice([36, 48, 60, 72]))
            rate = Decimal(str(round(float(rng.uniform(8.0, 12.0)), 2)))
        elif loan_type == "education":
            principal = Decimal(str(round(float(rng.uniform(2_00_000, 20_00_000)), 2)))
            tenure = int(rng.choice([60, 84, 120]))
            rate = Decimal(str(round(float(rng.uniform(8.5, 11.5)), 2)))
        else:
            principal = Decimal(str(round(float(rng.uniform(50_000, 10_00_000)), 2)))
            tenure = int(rng.choice([12, 24, 36, 48, 60]))
            rate = Decimal(str(round(float(rng.uniform(10.0, 18.0)), 2)))

        monthly_rate = float(rate) / 100 / 12
        if monthly_rate > 0:
            emi_val = float(principal) * monthly_rate * (1 + monthly_rate) ** tenure / (
                (1 + monthly_rate) ** tenure - 1
            )
        else:
            emi_val = float(principal) / tenure
        emi = Decimal(str(round(emi_val, 2)))

        credit_score = int(rng.integers(580, 850))

        status_roll = rng.random()
        status = "active" if status_roll < 0.75 else ("closed" if status_roll < 0.92 else "defaulted")

        disbursed = cust.created_at + timedelta(days=int(rng.integers(30, 365)))

        loan = Loan(
            loan_uid=str(uuid.UUID(int=int(rng.integers(0, 2**63)) ^ (int(rng.integers(0, 2**63)) << 64))),
            customer_id=cust.id,
            loan_type=loan_type,
            principal=principal,
            interest_rate=rate,
            tenure_months=tenure,
            emi=emi,
            status=status,
            disbursed_at=disbursed,
            credit_score=credit_score,
        )
        loans.append(loan)

    session.add_all(loans)
    session.flush()
    return loans


def seed_database(
    database_url: str | None = None,
    seed: int = SEED,
    num_customers: int = NUM_CUSTOMERS,
    target_transactions: int = TARGET_TRANSACTIONS,
) -> dict:
    """Generate the full synthetic dataset. Returns row-count summary.

    Idempotent: drops and recreates all tables on each run so the seed
    produces identical data regardless of prior state.
    """
    engine = init_db(database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    fake = _seeded_faker(seed)
    rng = _seeded_rng(seed)

    session = get_session(engine)
    try:
        customers = generate_customers(session, fake, rng, n=num_customers)
        accounts = generate_accounts(session, rng, customers)
        n_txns = generate_transactions(session, fake, rng, accounts, target=target_transactions)
        loans = generate_loans(session, rng, customers)
        session.commit()

        return {
            "customers": len(customers),
            "accounts": len(accounts),
            "transactions": n_txns,
            "loans": len(loans),
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    counts = seed_database()
    print("Synthetic data generated:")
    for table, count in counts.items():
        print(f"  {table}: {count:,}")
