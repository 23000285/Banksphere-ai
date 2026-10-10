"""Base agent with step limits, error handling, and structured tracing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class AgentStep(BaseModel):
    step: int
    tool_name: str
    outcome: str  # "success", "error", "denied"
    result_summary: str
    duration_ms: float = 0.0


class AgentResult(BaseModel):
    agent: str
    trace_id: str
    status: str  # "completed", "error", "max_steps_reached"
    steps: list[AgentStep]
    response: dict[str, Any] = {}
    error: str | None = None
    is_mock: bool = True


MAX_STEPS = 10


@dataclass
class BaseAgent:
    name: str
    max_steps: int = MAX_STEPS
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _steps: list[AgentStep] = field(default_factory=list, repr=False)
    _step_count: int = field(default=0, repr=False)

    def _record_step(
        self,
        tool_name: str,
        outcome: str,
        summary: str,
        duration_ms: float = 0.0,
    ) -> AgentStep:
        self._step_count += 1
        step = AgentStep(
            step=self._step_count,
            tool_name=tool_name,
            outcome=outcome,
            result_summary=summary,
            duration_ms=round(duration_ms, 2),
        )
        self._steps.append(step)
        return step

    def _make_result(
        self,
        status: str,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AgentResult:
        return AgentResult(
            agent=self.name,
            trace_id=self.trace_id,
            status=status,
            steps=list(self._steps),
            response=response or {},
            error=error,
        )

    def _check_step_limit(self) -> AgentResult | None:
        if self._step_count >= self.max_steps:
            return self._make_result(
                status="max_steps_reached",
                error=f"Agent '{self.name}' reached step limit ({self.max_steps})",
            )
        return None
