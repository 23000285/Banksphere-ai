"""Supervisor agent — LangGraph StateGraph routing to specialist agents.

Classifies intent, dispatches to the correct specialist (banking, fraud,
credit), aggregates results, and handles unsupported/ambiguous requests.
Execution is sequential. The supervisor itself has no direct tool access.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.banking_agent import BankingAgent
from agents.credit_agent import CreditAgent
from agents.fraud_agent import FraudAgent
from ml.credit_model import CreditModelArtifacts
from ml.fraud_model import FraudModelArtifacts

MAX_SUPERVISOR_STEPS = 5


# ---------------------------------------------------------------------------
# Pydantic models for validated input/output
# ---------------------------------------------------------------------------

class RoutingRequest(BaseModel):
    query: str = Field(..., min_length=1)
    customer_uid: str = Field(..., min_length=1)
    loan_amount: str | None = None
    loan_tenure_months: int = 36
    credit_score: int = 700
    annual_income: str = "600000"
    employment_years: int = 3
    age: int = 30


class RoutingResult(BaseModel):
    query: str
    intent: str
    specialist: str
    status: str
    trace_id: str
    result: dict[str, Any] = {}
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal graph state (TypedDict for LangGraph compatibility)
# ---------------------------------------------------------------------------

class _GraphState(TypedDict, total=False):
    query: str
    customer_uid: str
    intent: str
    specialist: str
    result: dict
    status: str
    error: str
    trace_id: str
    steps_taken: int
    # Carried through but not part of graph logic
    request_params: dict


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("fraud", ["fraud", "fraudulent", "suspicious", "risk score", "flagged", "anomaly"]),
    ("credit", ["credit", "loan", "emi", "eligibility", "borrow", "interest rate"]),
    ("banking", ["balance", "transactions", "transaction", "account", "profile", "card", "transfer", "statement"]),
]


def classify_intent(query: str) -> str:
    lower = query.lower()
    for intent, keywords in _INTENT_PATTERNS:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                return intent
    return "unsupported"


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------

def _classify_node(state: _GraphState) -> dict:
    intent = classify_intent(state["query"])
    return {"intent": intent, "specialist": intent, "steps_taken": state.get("steps_taken", 0) + 1}


def _banking_node(state: _GraphState, session: Session) -> dict:
    agent = BankingAgent(trace_id=state.get("trace_id", ""))
    result = agent.get_balance_and_transactions(session, state["customer_uid"])
    return {
        "result": result.model_dump(mode="json"),
        "status": result.status,
        "error": result.error,
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


def _fraud_node(state: _GraphState, session: Session, fraud_artifacts: FraudModelArtifacts) -> dict:
    agent = FraudAgent(trace_id=state.get("trace_id", ""))
    result = agent.score_customer_transactions(
        session, state["customer_uid"], fraud_artifacts, txn_limit=5,
    )
    return {
        "result": result.model_dump(mode="json"),
        "status": result.status,
        "error": result.error,
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


def _credit_node(
    state: _GraphState,
    session: Session,
    credit_artifacts: CreditModelArtifacts,
) -> dict:
    from decimal import Decimal

    params = state.get("request_params", {})
    loan_amount = params.get("loan_amount", "500000")
    agent = CreditAgent(trace_id=state.get("trace_id", ""))
    result = agent.assess_loan_eligibility(
        session=session,
        customer_uid=state["customer_uid"],
        credit_artifacts=credit_artifacts,
        loan_amount=Decimal(str(loan_amount)),
        loan_tenure_months=params.get("loan_tenure_months", 36),
        credit_score=params.get("credit_score", 700),
        annual_income=Decimal(str(params.get("annual_income", "600000"))),
        employment_years=params.get("employment_years", 3),
        age=params.get("age", 30),
    )
    return {
        "result": result.model_dump(mode="json"),
        "status": result.status,
        "error": result.error,
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


def _unsupported_node(state: _GraphState) -> dict:
    return {
        "result": {},
        "status": "unsupported",
        "error": f"Could not route query: '{state['query']}'. Supported intents: banking, fraud, credit.",
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


def _route_condition(state: _GraphState) -> str:
    intent = state.get("intent", "unsupported")
    if intent in ("banking", "fraud", "credit"):
        return intent
    return "unsupported"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_supervisor_graph(
    session: Session,
    fraud_artifacts: FraudModelArtifacts | None = None,
    credit_artifacts: CreditModelArtifacts | None = None,
) -> StateGraph:
    graph = StateGraph(_GraphState)

    graph.add_node("classify", _classify_node)
    graph.add_node("banking", lambda s: _banking_node(s, session))
    graph.add_node("fraud", lambda s: _fraud_node(s, session, fraud_artifacts))
    graph.add_node("credit", lambda s: _credit_node(s, session, credit_artifacts))
    graph.add_node("unsupported", _unsupported_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        _route_condition,
        {
            "banking": "banking",
            "fraud": "fraud",
            "credit": "credit",
            "unsupported": "unsupported",
        },
    )
    graph.add_edge("banking", END)
    graph.add_edge("fraud", END)
    graph.add_edge("credit", END)
    graph.add_edge("unsupported", END)

    return graph


def compile_supervisor(
    session: Session,
    fraud_artifacts: FraudModelArtifacts | None = None,
    credit_artifacts: CreditModelArtifacts | None = None,
):
    graph = build_supervisor_graph(session, fraud_artifacts, credit_artifacts)
    return graph.compile()


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_supervisor(
    request: RoutingRequest,
    session: Session,
    fraud_artifacts: FraudModelArtifacts | None = None,
    credit_artifacts: CreditModelArtifacts | None = None,
) -> RoutingResult:
    trace_id = uuid.uuid4().hex[:12]

    app = compile_supervisor(session, fraud_artifacts, credit_artifacts)

    initial_state: _GraphState = {
        "query": request.query,
        "customer_uid": request.customer_uid,
        "intent": "",
        "specialist": "",
        "result": {},
        "status": "pending",
        "error": "",
        "trace_id": trace_id,
        "steps_taken": 0,
        "request_params": {
            "loan_amount": request.loan_amount or "500000",
            "loan_tenure_months": request.loan_tenure_months,
            "credit_score": request.credit_score,
            "annual_income": request.annual_income,
            "employment_years": request.employment_years,
            "age": request.age,
        },
    }

    final_state = app.invoke(initial_state)

    return RoutingResult(
        query=request.query,
        intent=final_state.get("intent", "unknown"),
        specialist=final_state.get("specialist", "none"),
        status=final_state.get("status", "error"),
        trace_id=trace_id,
        result=final_state.get("result", {}),
        error=final_state.get("error") or None,
    )
