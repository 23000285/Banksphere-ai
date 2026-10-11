"""Compliance tools — policy retrieval backed by hybrid RAG.

Registered as 'retrieve_policy' in the central tool registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag.hybrid_retriever import HybridRetriever, RetrievalResult

_retriever: HybridRetriever | None = None


class PolicyQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Policy question")
    top_k: int = Field(default=5, ge=1, le=20)


class PolicySource(BaseModel):
    chunk_id: str
    source: str
    category: str
    heading: str
    rrf_score: float
    chroma_rank: int | None = None
    bm25_rank: int | None = None


class PolicyQueryResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    sources: list[PolicySource]
    total_results: int
    has_evidence: bool


def init_retriever(
    chroma_persist_dir: Path | None = None,
    policies_dir: Path | None = None,
) -> HybridRetriever:
    """Initialize the module-level retriever (call once at startup)."""
    global _retriever
    _retriever = HybridRetriever(
        chroma_persist_dir=chroma_persist_dir,
        policies_dir=policies_dir,
    )
    return _retriever


def get_retriever() -> HybridRetriever | None:
    return _retriever


def retrieve_policy(request: PolicyQueryRequest) -> PolicyQueryResponse:
    """Retrieve policy evidence for a query using hybrid retrieval."""
    if _retriever is None:
        return PolicyQueryResponse(
            query=request.query,
            results=[],
            sources=[],
            total_results=0,
            has_evidence=False,
        )

    hits: list[RetrievalResult] = _retriever.query(
        request.query,
        top_k=request.top_k,
    )

    results: list[dict[str, Any]] = []
    sources: list[PolicySource] = []
    for h in hits:
        results.append(
            {
                "chunk_id": h.chunk_id,
                "text": h.text,
                "source": h.source,
                "category": h.category,
                "heading": h.heading,
            }
        )
        sources.append(
            PolicySource(
                chunk_id=h.chunk_id,
                source=h.source,
                category=h.category,
                heading=h.heading,
                rrf_score=h.rrf_score,
                chroma_rank=h.chroma_rank,
                bm25_rank=h.bm25_rank,
            )
        )

    return PolicyQueryResponse(
        query=request.query,
        results=results,
        sources=sources,
        total_results=len(results),
        has_evidence=len(results) > 0,
    )
