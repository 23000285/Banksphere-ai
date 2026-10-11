"""FastAPI entry point for BankSphere AI.

Exposes the multi-agent banking workflow through a REST API with
Pydantic validation, API-key authentication, trace IDs, and safe errors.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from gateway.pii import mask_pii

logger = logging.getLogger("banksphere.api")

# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, seed data, and train models once at startup."""
    settings = get_settings()
    logger.info("Starting BankSphere AI (env=%s)", settings.app_env)

    from ml.db_engine import init_db, get_session
    from ml.seed_data import seed_database
    from ml.fraud_model import train_fraud_models
    from ml.credit_model import train_credit_model
    from tools.registry import (
        register_banking_tools,
        register_fraud_tools,
        register_credit_tools,
        register_compliance_tools,
    )

    engine = init_db(settings.database_url)
    seed_database(settings.database_url)

    _state["engine"] = engine
    _state["fraud_artifacts"] = train_fraud_models()
    _state["credit_artifacts"] = train_credit_model()

    from tools.compliance_tools import init_retriever
    from rag.chroma_store import ingest_chunks
    from rag.loader import load_policy_documents

    chunks = load_policy_documents()
    if chunks:
        ingest_chunks(chunks)

    register_banking_tools()
    register_fraud_tools()
    register_credit_tools()
    register_compliance_tools()
    init_retriever()

    logger.info("Startup complete — models trained, tools registered")
    yield
    _state.clear()


app = FastAPI(
    title="BankSphere AI",
    description="Multi-agent banking MVP API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Validate the API key from the X-API-Key header."""
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    settings = get_settings()
    if settings.api_secret_key in ("", "CHANGE_ME_BEFORE_USE"):
        raise HTTPException(
            status_code=503,
            detail="API key not configured. Set API_SECRET_KEY in .env.",
        )
    if api_key != settings.api_secret_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ---------------------------------------------------------------------------
# DB session dependency
# ---------------------------------------------------------------------------

def get_db_session():
    from ml.db_engine import get_session
    session = get_session(_state["engine"])
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    customer_uid: str = Field(..., min_length=1)
    loan_amount: str | None = None
    loan_tenure_months: int = 36
    credit_score: int = 700
    annual_income: str = "600000"
    employment_years: int = 3
    age: int = 30


class ChatResponse(BaseModel):
    trace_id: str
    query: str
    intent: str
    specialist: str
    status: str
    result: dict[str, Any] = {}
    error: str | None = None


class ApprovalActionRequest(BaseModel):
    approved: bool
    resolved_by: str = "human_operator"
    customer_uid: str | None = None


class ApprovalResponse(BaseModel):
    request_id: str
    action: str
    customer_uid: str
    status: str
    resolved_by: str | None = None
    resolved_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Middleware — attach trace ID to every request
# ---------------------------------------------------------------------------

@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:12])
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    return HealthResponse(status="ok", version="0.1.0")


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    tags=["agent"],
)
async def chat(
    body: ChatRequest,
    request: Request,
    _key: str = Depends(verify_api_key),
    session: Session = Depends(get_db_session),
):
    """Route a user query through the supervisor to the appropriate specialist agent."""
    from agents.supervisor import RoutingRequest, run_supervisor

    trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex[:12])

    try:
        routing_req = RoutingRequest(
            query=body.query,
            customer_uid=body.customer_uid,
            loan_amount=body.loan_amount,
            loan_tenure_months=body.loan_tenure_months,
            credit_score=body.credit_score,
            annual_income=body.annual_income,
            employment_years=body.employment_years,
            age=body.age,
        )

        result = run_supervisor(
            routing_req,
            session=session,
            fraud_artifacts=_state.get("fraud_artifacts"),
            credit_artifacts=_state.get("credit_artifacts"),
        )

        safe_error = mask_pii(result.error) if result.error else None

        return ChatResponse(
            trace_id=result.trace_id,
            query=body.query,
            intent=result.intent,
            specialist=result.specialist,
            status=result.status,
            result=result.result,
            error=safe_error,
        )

    except Exception:
        logger.exception("Unhandled error in /chat (trace=%s)", trace_id)
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check trace ID for details.",
            headers={"X-Trace-ID": trace_id},
        )


@app.post(
    "/approval/{request_id}",
    response_model=ApprovalResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["approval"],
)
async def resolve_approval_endpoint(
    request_id: str,
    body: ApprovalActionRequest,
    _key: str = Depends(verify_api_key),
):
    """Approve or reject a pending sensitive-action request."""
    from gateway.approval import get_approval_log, resolve_approval

    if body.customer_uid:
        matching = [
            r for r in get_approval_log()
            if r.request_id == request_id
        ]
        if matching and matching[0].customer_uid != body.customer_uid:
            raise HTTPException(
                status_code=403,
                detail="Cannot resolve approval for a different customer.",
            )

    resolved = resolve_approval(
        request_id=request_id,
        approved=body.approved,
        resolved_by=body.resolved_by,
    )
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{request_id}' not found or already resolved.",
        )

    return ApprovalResponse(
        request_id=resolved.request_id,
        action=resolved.action,
        customer_uid=resolved.customer_uid,
        status=resolved.status.value,
        resolved_by=resolved.resolved_by,
        resolved_at=resolved.resolved_at,
    )


@app.get(
    "/approvals/pending",
    response_model=list[ApprovalResponse],
    tags=["approval"],
)
async def list_pending_approvals(
    customer_uid: str | None = None,
    _key: str = Depends(verify_api_key),
):
    """List all pending approval requests, optionally filtered by customer."""
    from gateway.approval import get_pending_approvals

    pending = get_pending_approvals(customer_uid=customer_uid)
    return [
        ApprovalResponse(
            request_id=r.request_id,
            action=r.action,
            customer_uid=r.customer_uid,
            status=r.status.value,
            resolved_by=r.resolved_by,
            resolved_at=r.resolved_at,
        )
        for r in pending
    ]
