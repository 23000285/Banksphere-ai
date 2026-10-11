"""Human approval gate for sensitive actions.

Sensitive actions (card block, fund transfer, account closure) must not
be auto-approved by an agent or model recommendation.  This module
provides a structured approval workflow that pauses execution and
requires explicit human confirmation before proceeding.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger("banksphere.approval")

APPROVAL_EXPIRY_MINUTES = 15


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


SENSITIVE_ACTIONS = frozenset({
    "card_block",
    "card_unblock",
    "fund_transfer",
    "account_closure",
    "limit_increase",
})


@dataclass
class ApprovalRequest:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: str = ""
    customer_uid: str = ""
    agent: str = ""
    trace_id: str = ""
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: str = field(
        default_factory=lambda: (
            datetime.now(timezone.utc) + timedelta(minutes=APPROVAL_EXPIRY_MINUTES)
        ).isoformat()
    )
    resolved_at: str | None = None
    resolved_by: str | None = None


_approval_log: list[ApprovalRequest] = []


def is_sensitive_action(action: str) -> bool:
    """Return True if the action requires human approval."""
    return action in SENSITIVE_ACTIONS


def request_approval(
    action: str,
    customer_uid: str,
    agent: str,
    trace_id: str = "",
    reason: str = "",
    details: dict[str, Any] | None = None,
) -> ApprovalRequest:
    """Create a pending approval request for a sensitive action.

    The request is logged and stored.  Execution must NOT proceed until
    the request is explicitly approved via ``resolve_approval``.
    """
    req = ApprovalRequest(
        action=action,
        customer_uid=customer_uid,
        agent=agent,
        trace_id=trace_id or uuid.uuid4().hex[:12],
        reason=reason,
        details=details or {},
    )
    _approval_log.append(req)
    logger.info(
        "APPROVAL_REQUESTED %s",
        {
            "request_id": req.request_id,
            "action": req.action,
            "customer_uid": req.customer_uid,
            "agent": req.agent,
            "trace_id": req.trace_id,
            "status": req.status.value,
        },
    )
    return req


def _check_expiry(req: ApprovalRequest) -> bool:
    """Mark the request as EXPIRED if past its deadline. Returns True if expired."""
    if req.status != ApprovalStatus.PENDING:
        return req.status == ApprovalStatus.EXPIRED
    now = datetime.now(timezone.utc)
    expires = datetime.fromisoformat(req.expires_at)
    if now >= expires:
        req.status = ApprovalStatus.EXPIRED
        req.resolved_at = now.isoformat()
        req.resolved_by = "system_expiry"
        logger.info(
            "APPROVAL_EXPIRED %s",
            {"request_id": req.request_id, "action": req.action},
        )
        return True
    return False


def resolve_approval(
    request_id: str,
    approved: bool,
    resolved_by: str = "human_operator",
) -> ApprovalRequest | None:
    """Approve or reject a pending approval request.

    Returns the updated request, or None if not found / already resolved / expired.
    """
    for req in _approval_log:
        if req.request_id == request_id:
            if _check_expiry(req):
                return None
            if req.status != ApprovalStatus.PENDING:
                return None
            req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.resolved_by = resolved_by
            logger.info(
                "APPROVAL_RESOLVED %s",
                {
                    "request_id": req.request_id,
                    "action": req.action,
                    "status": req.status.value,
                    "resolved_by": resolved_by,
                },
            )
            return req
    return None


def get_pending_approvals(customer_uid: str | None = None) -> list[ApprovalRequest]:
    """Return all pending approval requests, optionally filtered by customer.

    Expired requests are marked as EXPIRED before filtering.
    """
    for r in _approval_log:
        if r.status == ApprovalStatus.PENDING:
            _check_expiry(r)
    return [
        r for r in _approval_log
        if r.status == ApprovalStatus.PENDING
        and (customer_uid is None or r.customer_uid == customer_uid)
    ]


def get_approval_log() -> list[ApprovalRequest]:
    """Return all approval requests (useful for testing and audit)."""
    return list(_approval_log)


def clear_approval_log() -> None:
    """Clear the in-memory approval log."""
    _approval_log.clear()
