"""Central tool registry with per-agent permission enforcement.

Each agent is allowed to call only its permitted tools. Any attempt to invoke
a tool outside the agent's permission set is denied before execution.
"""

from __future__ import annotations

from typing import Any, Callable

from gateway.audit import AuditEvent


class ToolPermissionDenied(Exception):
    pass


# Maps agent name -> set of allowed tool names
AGENT_PERMISSIONS: dict[str, set[str]] = {
    "banking": {
        "get_customer_profile",
        "get_accounts",
        "get_transactions",
        "get_card_status",
    },
    "fraud": {
        "get_transactions",
        "score_transaction",
    },
    "credit": {
        "get_customer_profile",
        "assess_credit_risk",
        "calculate_emi",
    },
    "compliance": {
        "retrieve_policy",
    },
    "supervisor": set(),  # routing only — no direct tool access
}


_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_tool(name: str, func: Callable[..., Any]) -> None:
    """Register a tool function by name."""
    _TOOL_REGISTRY[name] = func


def get_tool(name: str) -> Callable[..., Any] | None:
    """Look up a registered tool by name."""
    return _TOOL_REGISTRY.get(name)


def list_tools() -> list[str]:
    """Return all registered tool names."""
    return sorted(_TOOL_REGISTRY.keys())


def agent_allowed_tools(agent: str) -> set[str]:
    """Return the set of tool names an agent is allowed to call."""
    return AGENT_PERMISSIONS.get(agent, set())


def invoke_tool(
    tool_name: str,
    agent: str,
    **kwargs: Any,
) -> Any:
    """Invoke a tool with permission enforcement.

    Raises ToolPermissionDenied if the agent is not allowed to call this tool.
    Raises KeyError if the tool is not registered.
    """
    allowed = agent_allowed_tools(agent)
    if tool_name not in allowed:
        evt = AuditEvent(
            tool_name=tool_name,
            agent=agent,
            requesting_customer_uid=kwargs.get("requesting_customer_uid", "unknown"),
            target_customer_uid=kwargs.get("target_customer_uid", "unknown"),
            outcome="denied",
            detail=f"agent '{agent}' is not permitted to call '{tool_name}'",
        )
        evt.log()
        raise ToolPermissionDenied(
            f"Agent '{agent}' is not permitted to call tool '{tool_name}'. "
            f"Allowed tools: {sorted(allowed)}"
        )

    func = _TOOL_REGISTRY.get(tool_name)
    if func is None:
        raise KeyError(f"Tool '{tool_name}' is not registered")

    return func(**kwargs)


def register_banking_tools() -> None:
    """Register all banking domain tools."""
    from tools.banking_tools import (
        get_accounts,
        get_card_status,
        get_customer_profile,
        get_transactions,
    )

    register_tool("get_customer_profile", get_customer_profile)
    register_tool("get_accounts", get_accounts)
    register_tool("get_transactions", get_transactions)
    register_tool("get_card_status", get_card_status)
