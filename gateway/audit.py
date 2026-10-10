"""Structured audit logging — records tool invocations without secrets or PII."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("banksphere.audit")


@dataclass
class AuditEvent:
    tool_name: str
    agent: str
    requesting_customer_uid: str
    target_customer_uid: str
    outcome: str  # "success", "denied", "error", "not_found"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    duration_ms: float = 0.0
    detail: str = ""

    def log(self) -> dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "tool": self.tool_name,
            "agent": self.agent,
            "requester": self.requesting_customer_uid,
            "target": self.target_customer_uid,
            "outcome": self.outcome,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.detail:
            record["detail"] = self.detail

        if self.outcome == "denied":
            logger.warning("AUDIT %s", record)
        elif self.outcome == "error":
            logger.error("AUDIT %s", record)
        else:
            logger.info("AUDIT %s", record)

        return record


class AuditTimer:
    """Context manager that measures elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> AuditTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
