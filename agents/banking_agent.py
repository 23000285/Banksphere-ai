"""Banking agent — balance, transactions, profile, card status.

Calls only tools in the 'banking' permission set via the central registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from agents.base import AgentResult, BaseAgent
from gateway.audit import AuditTimer
from tools.banking_tools import AuthorizationDenied, CustomerNotFound
from tools.registry import ToolPermissionDenied, invoke_tool
from tools.schemas import (
    AccountsRequest,
    CardBlockRequest,
    CustomerProfileRequest,
    TransactionsRequest,
)

AGENT_NAME = "banking"


@dataclass
class BankingAgent(BaseAgent):
    name: str = AGENT_NAME

    def get_balance_and_transactions(
        self,
        session: Session,
        customer_uid: str,
        txn_limit: int = 5,
    ) -> AgentResult:
        accounts_resp = self._call_tool(
            "get_accounts",
            session=session,
            request=AccountsRequest(
                customer_uid=customer_uid,
                requesting_customer_uid=customer_uid,
            ),
            agent=AGENT_NAME,
        )
        if accounts_resp is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        limit_hit = self._check_step_limit()
        if limit_hit:
            return limit_hit

        txns_resp = self._call_tool(
            "get_transactions",
            session=session,
            request=TransactionsRequest(
                customer_uid=customer_uid,
                requesting_customer_uid=customer_uid,
                limit=txn_limit,
            ),
            agent=AGENT_NAME,
        )
        if txns_resp is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        return self._make_result(
            "completed",
            response={
                "accounts": [a.model_dump(mode="json") for a in accounts_resp.accounts],
                "transactions": [t.model_dump(mode="json") for t in txns_resp.transactions],
                "total_transactions_returned": txns_resp.total_returned,
            },
        )

    def handle_card_block(
        self,
        session: Session,
        customer_uid: str,
        account_number: str | None = None,
        reason: str = "Customer requested card block",
    ) -> AgentResult:
        if account_number is None:
            accounts_resp = self._call_tool(
                "get_accounts",
                session=session,
                request=AccountsRequest(
                    customer_uid=customer_uid,
                    requesting_customer_uid=customer_uid,
                ),
                agent=AGENT_NAME,
            )
            if accounts_resp is None:
                return self._make_result("error", error=self._steps[-1].result_summary)
            if not accounts_resp.accounts:
                return self._make_result("error", error="No accounts found for this customer.")
            account_number = accounts_resp.accounts[0].account_number

        block_resp = self._call_tool(
            "request_card_block",
            session=session,
            request=CardBlockRequest(
                customer_uid=customer_uid,
                requesting_customer_uid=customer_uid,
                account_number=account_number,
                reason=reason,
            ),
            agent=AGENT_NAME,
        )
        if block_resp is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        return self._make_result(
            "completed",
            response={
                "action": "card_block",
                "account_number": block_resp.account_number,
                "status": block_resp.status,
                "approval_request_id": block_resp.approval_request_id,
                "message": block_resp.message,
                "requires_approval": block_resp.requires_approval,
            },
        )

    def get_customer_profile(
        self,
        session: Session,
        customer_uid: str,
    ) -> AgentResult:
        profile = self._call_tool(
            "get_customer_profile",
            session=session,
            request=CustomerProfileRequest(
                customer_uid=customer_uid,
                requesting_customer_uid=customer_uid,
            ),
            agent=AGENT_NAME,
        )
        if profile is None:
            return self._make_result("error", error=self._steps[-1].result_summary)

        return self._make_result(
            "completed",
            response=profile.model_dump(mode="json"),
        )

    def _call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        limit_hit = self._check_step_limit()
        if limit_hit:
            return None

        with AuditTimer() as timer:
            try:
                result = invoke_tool(tool_name, **kwargs)
            except (AuthorizationDenied, ToolPermissionDenied) as exc:
                self._record_step(tool_name, "denied", str(exc), timer.elapsed_ms)
                return None
            except CustomerNotFound as exc:
                self._record_step(tool_name, "error", str(exc), timer.elapsed_ms)
                return None
            except Exception as exc:
                self._record_step(tool_name, "error", str(exc), timer.elapsed_ms)
                return None

        self._record_step(
            tool_name, "success", f"{tool_name} returned successfully", timer.elapsed_ms,
        )
        return result
