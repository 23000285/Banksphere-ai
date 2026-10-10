"""Fraud agent — transaction scoring and risk recommendation.

Calls only tools in the 'fraud' permission set. Returns allow/verify/block
recommendations but never auto-executes sensitive actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import Session

from agents.base import AgentResult, BaseAgent
from gateway.audit import AuditTimer
from ml.fraud_model import (
    FraudModelArtifacts,
    TransactionScoreRequest,
    TransactionScoreResponse,
    score_transaction,
)
from tools.banking_tools import AuthorizationDenied, CustomerNotFound
from tools.registry import ToolPermissionDenied, invoke_tool
from tools.schemas import TransactionsRequest

AGENT_NAME = "fraud"

_RECOMMENDATION_MAP = {
    "low": "allow",
    "medium": "verify",
    "high": "block",
}


@dataclass
class FraudAgent(BaseAgent):
    name: str = AGENT_NAME

    def score_customer_transactions(
        self,
        session: Session,
        customer_uid: str,
        fraud_artifacts: FraudModelArtifacts,
        txn_limit: int = 5,
    ) -> AgentResult:
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

        scored: list[dict[str, Any]] = []
        for txn in txns_resp.transactions:
            limit_hit = self._check_step_limit()
            if limit_hit:
                return limit_hit

            score_result = self._score_single(txn, fraud_artifacts)
            if score_result is None:
                continue
            scored.append(score_result)

        return self._make_result(
            "completed",
            response={
                "customer_uid": customer_uid,
                "scored_transactions": scored,
                "total_scored": len(scored),
            },
        )

    def score_single_transaction(
        self,
        request: TransactionScoreRequest,
        fraud_artifacts: FraudModelArtifacts,
    ) -> AgentResult:
        limit_hit = self._check_step_limit()
        if limit_hit:
            return limit_hit

        with AuditTimer() as timer:
            try:
                resp = score_transaction(request, fraud_artifacts)
            except RuntimeError as exc:
                self._record_step("score_transaction", "error", str(exc), timer.elapsed_ms)
                return self._make_result("error", error=str(exc))
            except Exception as exc:
                self._record_step("score_transaction", "error", str(exc), timer.elapsed_ms)
                return self._make_result("error", error=str(exc))

        recommendation = _RECOMMENDATION_MAP.get(resp.risk_category, "verify")
        self._record_step(
            "score_transaction",
            "success",
            f"fraud_score={resp.fraud_score:.4f} risk={resp.risk_category} rec={recommendation}",
            timer.elapsed_ms,
        )

        return self._make_result(
            "completed",
            response={
                "fraud_score": resp.fraud_score,
                "anomaly_score": resp.anomaly_score,
                "risk_category": resp.risk_category,
                "recommendation": recommendation,
                "requires_approval": recommendation == "block",
                "top_features": resp.top_features,
                "explanation_method": resp.explanation_method,
                "is_synthetic": resp.is_synthetic,
            },
        )

    def _score_single(
        self,
        txn: Any,
        fraud_artifacts: FraudModelArtifacts,
    ) -> dict[str, Any] | None:
        with AuditTimer() as timer:
            try:
                request = TransactionScoreRequest(
                    amount=txn.amount,
                    hour=txn.timestamp.hour,
                    channel=txn.channel,
                )
                resp = score_transaction(request, fraud_artifacts)
            except Exception as exc:
                self._record_step(
                    "score_transaction", "error",
                    f"txn {txn.transaction_uid}: {exc}", timer.elapsed_ms,
                )
                return None

        recommendation = _RECOMMENDATION_MAP.get(resp.risk_category, "verify")
        self._record_step(
            "score_transaction",
            "success",
            f"txn {txn.transaction_uid}: score={resp.fraud_score:.4f} rec={recommendation}",
            timer.elapsed_ms,
        )

        return {
            "transaction_uid": txn.transaction_uid,
            "amount": str(txn.amount),
            "fraud_score": resp.fraud_score,
            "anomaly_score": resp.anomaly_score,
            "risk_category": resp.risk_category,
            "recommendation": recommendation,
            "requires_approval": recommendation == "block",
            "top_features": resp.top_features,
        }

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
