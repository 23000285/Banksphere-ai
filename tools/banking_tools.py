"""Banking domain tools — customer profile, accounts, transactions, card status.

Every tool enforces customer-level authorization: the requesting_customer_uid
must match the target customer_uid. Cross-customer access is denied at the tool
level, not just in prompts or the UI.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.audit import AuditEvent, AuditTimer
from ml.db_models import Account, Customer, Transaction
from gateway.approval import is_sensitive_action, request_approval
from tools.schemas import (
    AccountSummary,
    AccountsRequest,
    AccountsResponse,
    CardBlockRequest,
    CardBlockResponse,
    CardStatusRequest,
    CardStatusResponse,
    CustomerProfileRequest,
    CustomerProfileResponse,
    ToolError,
    TransactionSummary,
    TransactionsRequest,
    TransactionsResponse,
)


class AuthorizationDenied(Exception):
    pass


class CustomerNotFound(Exception):
    pass


def _authorize(req_uid: str, target_uid: str, tool_name: str, agent: str) -> AuditEvent | None:
    """Return an AuditEvent with outcome='denied' if cross-customer access, else None."""
    if req_uid != target_uid:
        evt = AuditEvent(
            tool_name=tool_name,
            agent=agent,
            requesting_customer_uid=req_uid,
            target_customer_uid=target_uid,
            outcome="denied",
            detail="cross-customer access attempt",
        )
        evt.log()
        return evt
    return None


def _find_customer(session: Session, customer_uid: str) -> Customer | None:
    return session.execute(
        select(Customer).where(Customer.customer_uid == customer_uid)
    ).scalar_one_or_none()


def get_customer_profile(
    session: Session,
    request: CustomerProfileRequest,
    agent: str = "banking",
) -> CustomerProfileResponse | ToolError:
    denied = _authorize(
        request.requesting_customer_uid, request.customer_uid,
        "get_customer_profile", agent,
    )
    if denied:
        raise AuthorizationDenied(denied.detail)

    with AuditTimer() as timer:
        customer = _find_customer(session, request.customer_uid)

    if customer is None:
        AuditEvent(
            tool_name="get_customer_profile", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="not_found", duration_ms=timer.elapsed_ms,
        ).log()
        raise CustomerNotFound(f"No customer with UID {request.customer_uid}")

    account_count = len(customer.accounts)

    AuditEvent(
        tool_name="get_customer_profile", agent=agent,
        requesting_customer_uid=request.requesting_customer_uid,
        target_customer_uid=request.customer_uid,
        outcome="success", duration_ms=timer.elapsed_ms,
    ).log()

    return CustomerProfileResponse(
        customer_uid=customer.customer_uid,
        first_name=customer.first_name,
        last_name=customer.last_name,
        email=customer.email,
        phone=customer.phone,
        city=customer.city,
        state=customer.state,
        kyc_status=customer.kyc_status,
        risk_category=customer.risk_category,
        account_count=account_count,
    )


def get_accounts(
    session: Session,
    request: AccountsRequest,
    agent: str = "banking",
) -> AccountsResponse | ToolError:
    denied = _authorize(
        request.requesting_customer_uid, request.customer_uid,
        "get_accounts", agent,
    )
    if denied:
        raise AuthorizationDenied(denied.detail)

    with AuditTimer() as timer:
        customer = _find_customer(session, request.customer_uid)

    if customer is None:
        AuditEvent(
            tool_name="get_accounts", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="not_found", duration_ms=timer.elapsed_ms,
        ).log()
        raise CustomerNotFound(f"No customer with UID {request.customer_uid}")

    accounts = [
        AccountSummary(
            account_number=a.account_number,
            account_type=a.account_type,
            balance=a.balance,
            currency=a.currency,
            status=a.status,
            opened_at=a.opened_at,
        )
        for a in customer.accounts
    ]

    AuditEvent(
        tool_name="get_accounts", agent=agent,
        requesting_customer_uid=request.requesting_customer_uid,
        target_customer_uid=request.customer_uid,
        outcome="success", duration_ms=timer.elapsed_ms,
    ).log()

    return AccountsResponse(customer_uid=customer.customer_uid, accounts=accounts)


def get_transactions(
    session: Session,
    request: TransactionsRequest,
    agent: str = "banking",
) -> TransactionsResponse | ToolError:
    denied = _authorize(
        request.requesting_customer_uid, request.customer_uid,
        "get_transactions", agent,
    )
    if denied:
        raise AuthorizationDenied(denied.detail)

    with AuditTimer() as timer:
        customer = _find_customer(session, request.customer_uid)

    if customer is None:
        AuditEvent(
            tool_name="get_transactions", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="not_found", duration_ms=timer.elapsed_ms,
        ).log()
        raise CustomerNotFound(f"No customer with UID {request.customer_uid}")

    account_ids = [a.id for a in customer.accounts]
    if not account_ids:
        AuditEvent(
            tool_name="get_transactions", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="success", duration_ms=timer.elapsed_ms,
        ).log()
        return TransactionsResponse(
            customer_uid=customer.customer_uid, transactions=[], total_returned=0,
        )

    acct_map = {a.id: a.account_number for a in customer.accounts}

    rows = session.execute(
        select(Transaction)
        .where(Transaction.account_id.in_(account_ids))
        .order_by(Transaction.timestamp.desc())
        .limit(request.limit)
    ).scalars().all()

    txns = [
        TransactionSummary(
            transaction_uid=t.transaction_uid,
            account_number=acct_map[t.account_id],
            txn_type=t.txn_type,
            category=t.category,
            amount=t.amount,
            merchant=t.merchant,
            channel=t.channel,
            timestamp=t.timestamp,
            is_flagged=t.is_flagged,
        )
        for t in rows
    ]

    AuditEvent(
        tool_name="get_transactions", agent=agent,
        requesting_customer_uid=request.requesting_customer_uid,
        target_customer_uid=request.customer_uid,
        outcome="success", duration_ms=timer.elapsed_ms,
    ).log()

    return TransactionsResponse(
        customer_uid=customer.customer_uid, transactions=txns, total_returned=len(txns),
    )


def get_card_status(
    session: Session,
    request: CardStatusRequest,
    agent: str = "banking",
) -> CardStatusResponse | ToolError:
    denied = _authorize(
        request.requesting_customer_uid, request.customer_uid,
        "get_card_status", agent,
    )
    if denied:
        raise AuthorizationDenied(denied.detail)

    with AuditTimer() as timer:
        customer = _find_customer(session, request.customer_uid)

    if customer is None:
        AuditEvent(
            tool_name="get_card_status", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="not_found", duration_ms=timer.elapsed_ms,
        ).log()
        raise CustomerNotFound(f"No customer with UID {request.customer_uid}")

    account = next(
        (a for a in customer.accounts if a.account_number == request.account_number),
        None,
    )
    if account is None:
        AuditEvent(
            tool_name="get_card_status", agent=agent,
            requesting_customer_uid=request.requesting_customer_uid,
            target_customer_uid=request.customer_uid,
            outcome="not_found", duration_ms=timer.elapsed_ms,
            detail=f"account {request.account_number} not found for customer",
        ).log()
        raise CustomerNotFound(
            f"Account {request.account_number} not found for customer {request.customer_uid}"
        )

    status_messages = {
        "active": "Card is active and available for transactions.",
        "frozen": "Card is frozen. Contact support to unfreeze.",
        "closed": "This account is closed. Card is no longer valid.",
    }

    AuditEvent(
        tool_name="get_card_status", agent=agent,
        requesting_customer_uid=request.requesting_customer_uid,
        target_customer_uid=request.customer_uid,
        outcome="success", duration_ms=timer.elapsed_ms,
    ).log()

    return CardStatusResponse(
        customer_uid=customer.customer_uid,
        account_number=account.account_number,
        account_type=account.account_type,
        status=account.status,
        message=status_messages.get(account.status, "Unknown status."),
    )


def request_card_block(
    session: Session,
    request: CardBlockRequest,
    agent: str = "banking",
) -> CardBlockResponse:
    denied = _authorize(
        request.requesting_customer_uid, request.customer_uid,
        "request_card_block", agent,
    )
    if denied:
        raise AuthorizationDenied(denied.detail)

    customer = _find_customer(session, request.customer_uid)
    if customer is None:
        raise CustomerNotFound(f"No customer with UID {request.customer_uid}")

    account = next(
        (a for a in customer.accounts if a.account_number == request.account_number),
        None,
    )
    if account is None:
        raise CustomerNotFound(
            f"Account {request.account_number} not found for customer {request.customer_uid}"
        )

    if account.status == "frozen":
        return CardBlockResponse(
            customer_uid=customer.customer_uid,
            account_number=account.account_number,
            status="already_blocked",
            approval_request_id="",
            message="This card is already blocked.",
            requires_approval=False,
        )

    approval_req = request_approval(
        action="card_block",
        customer_uid=request.customer_uid,
        agent=agent,
        reason=request.reason,
        details={
            "account_number": request.account_number,
            "account_type": account.account_type,
        },
    )

    AuditEvent(
        tool_name="request_card_block", agent=agent,
        requesting_customer_uid=request.requesting_customer_uid,
        target_customer_uid=request.customer_uid,
        outcome="pending_approval",
        detail=f"Card block requested for {request.account_number}, approval_id={approval_req.request_id}",
    ).log()

    return CardBlockResponse(
        customer_uid=customer.customer_uid,
        account_number=account.account_number,
        status="pending_approval",
        approval_request_id=approval_req.request_id,
        message="Card block request submitted. Awaiting human approval before execution.",
        requires_approval=True,
    )


def execute_card_block(session: Session, approval_request_id: str) -> dict:
    """Execute a card block only if the corresponding approval is APPROVED.

    Returns a dict with status and message.  This function is the enforcement
    point — the card status change happens here, never before approval.
    """
    from gateway.approval import ApprovalStatus, get_approval_log

    matching = [
        r for r in get_approval_log()
        if r.request_id == approval_request_id and r.action == "card_block"
    ]
    if not matching:
        return {"status": "error", "message": "Approval request not found."}

    req = matching[0]

    if req.status == ApprovalStatus.PENDING:
        return {
            "status": "pending",
            "message": "Card block is still awaiting approval.",
            "customer_uid": req.customer_uid,
        }
    if req.status == ApprovalStatus.REJECTED:
        return {
            "status": "rejected",
            "message": "Card block was rejected by the operator.",
            "customer_uid": req.customer_uid,
        }
    if req.status == ApprovalStatus.EXPIRED:
        return {
            "status": "expired",
            "message": "Card block approval has expired.",
            "customer_uid": req.customer_uid,
        }
    if req.status != ApprovalStatus.APPROVED:
        return {"status": "error", "message": f"Unexpected approval status: {req.status}"}

    account_number = req.details.get("account_number", "")
    customer = _find_customer(session, req.customer_uid)
    if customer is None:
        return {"status": "error", "message": "Customer not found."}

    account = next(
        (a for a in customer.accounts if a.account_number == account_number),
        None,
    )
    if account is None:
        return {"status": "error", "message": f"Account {account_number} not found."}

    account.status = "frozen"
    session.commit()

    AuditEvent(
        tool_name="execute_card_block", agent="system",
        requesting_customer_uid=req.customer_uid,
        target_customer_uid=req.customer_uid,
        outcome="success",
        detail=f"Card blocked for account {account_number} (approval {approval_request_id})",
    ).log()

    return {
        "status": "blocked",
        "message": f"Card for account {account_number} has been blocked successfully.",
        "customer_uid": req.customer_uid,
        "account_number": account_number,
    }
