"""SQLAlchemy ORM models for the synthetic banking database."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_uid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    date_of_birth: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    address: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    pin_code: Mapped[str] = mapped_column(String(10), nullable=False)
    kyc_status: Mapped[str] = mapped_column(
        Enum("verified", "pending", "rejected", name="kyc_status_enum"),
        default="verified",
    )
    risk_category: Mapped[str] = mapped_column(
        Enum("low", "medium", "high", name="risk_category_enum"),
        default="low",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    accounts: Mapped[list[Account]] = relationship(back_populates="customer")
    loans: Mapped[list[Loan]] = relationship(back_populates="customer")


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    account_type: Mapped[str] = mapped_column(
        Enum("savings", "current", "salary", name="account_type_enum"),
        nullable=False,
    )
    balance: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    status: Mapped[str] = mapped_column(
        Enum("active", "frozen", "closed", name="account_status_enum"),
        default="active",
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="accounts")
    transactions: Mapped[list[Transaction]] = relationship(back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_uid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)
    txn_type: Mapped[str] = mapped_column(
        Enum("credit", "debit", name="txn_type_enum"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    merchant: Mapped[str] = mapped_column(String(200), nullable=True)
    channel: Mapped[str] = mapped_column(
        Enum("upi", "neft", "imps", "card", "atm", "branch", name="channel_enum"),
        nullable=False,
    )
    device_id: Mapped[str] = mapped_column(String(50), nullable=True)
    location_city: Mapped[str] = mapped_column(String(100), nullable=True)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    account: Mapped[Account] = relationship(back_populates="transactions")

    __table_args__ = (
        UniqueConstraint("transaction_uid", name="uq_transaction_uid"),
    )


class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loan_uid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    loan_type: Mapped[str] = mapped_column(
        Enum("personal", "home", "auto", "education", name="loan_type_enum"),
        nullable=False,
    )
    principal: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    interest_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    tenure_months: Mapped[int] = mapped_column(Integer, nullable=False)
    emi: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("active", "closed", "defaulted", name="loan_status_enum"),
        default="active",
    )
    disbursed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False)

    customer: Mapped[Customer] = relationship(back_populates="loans")
