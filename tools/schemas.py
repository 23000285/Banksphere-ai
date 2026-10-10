"""Pydantic input/output schemas for banking tools."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class CustomerProfileRequest(BaseModel):
    customer_uid: str = Field(..., description="Unique customer identifier")
    requesting_customer_uid: str = Field(..., description="UID of the authenticated caller")


class AccountsRequest(BaseModel):
    customer_uid: str
    requesting_customer_uid: str


class TransactionsRequest(BaseModel):
    customer_uid: str
    requesting_customer_uid: str
    limit: int = Field(default=5, ge=1, le=100)


class CardStatusRequest(BaseModel):
    customer_uid: str
    requesting_customer_uid: str
    account_number: str


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class CustomerProfileResponse(BaseModel):
    customer_uid: str
    first_name: str
    last_name: str
    email: str
    phone: str
    city: str
    state: str
    kyc_status: str
    risk_category: str
    account_count: int


class AccountSummary(BaseModel):
    account_number: str
    account_type: str
    balance: Decimal
    currency: str
    status: str
    opened_at: datetime


class AccountsResponse(BaseModel):
    customer_uid: str
    accounts: list[AccountSummary]


class TransactionSummary(BaseModel):
    transaction_uid: str
    account_number: str
    txn_type: str
    category: str
    amount: Decimal
    merchant: str | None
    channel: str
    timestamp: datetime
    is_flagged: bool


class TransactionsResponse(BaseModel):
    customer_uid: str
    transactions: list[TransactionSummary]
    total_returned: int


class CardStatusResponse(BaseModel):
    customer_uid: str
    account_number: str
    account_type: str
    status: str
    message: str


class ToolError(BaseModel):
    error: str
    code: str
