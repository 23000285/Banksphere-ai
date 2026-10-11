"""Notification stub — writes structured simulated events to a log.

This stub never sends real email, SMS, or push notifications.  Every call
produces a ``NotificationEvent`` that is logged via Python's logging module
and returned for programmatic inspection.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("banksphere.notifications")


@dataclass
class NotificationEvent:
    channel: str  # "email", "sms", "push", "in_app"
    recipient: str  # customer_uid or generic identifier
    subject: str
    body: str
    event_type: str  # "info", "alert", "approval_request", "policy_response"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    delivered: bool = False  # always False — this is a stub

    def log(self) -> dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "channel": self.channel,
            "recipient": self.recipient,
            "subject": self.subject,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "delivered": self.delivered,
            "simulated": True,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        logger.info("NOTIFICATION_STUB %s", record)
        return record


_event_log: list[NotificationEvent] = []


def send_notification(
    channel: str,
    recipient: str,
    subject: str,
    body: str,
    event_type: str = "info",
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NotificationEvent:
    """Create and log a simulated notification event.

    Returns the event for inspection.  Nothing is actually sent.
    """
    event = NotificationEvent(
        channel=channel,
        recipient=recipient,
        subject=subject,
        body=body,
        event_type=event_type,
        trace_id=trace_id or uuid.uuid4().hex[:12],
        metadata=metadata or {},
    )
    event.log()
    _event_log.append(event)
    return event


def get_event_log() -> list[NotificationEvent]:
    """Return all simulated notification events (useful for testing)."""
    return list(_event_log)


def clear_event_log() -> None:
    """Clear the in-memory event log."""
    _event_log.clear()
