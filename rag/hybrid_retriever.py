"""Hybrid retrieval combining ChromaDB vector search and BM25 keyword search.

Uses Reciprocal Rank Fusion (RRF) for deterministic, score-agnostic merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.bm25_retriever import BM25Retriever
from rag.chroma_store import query_chroma
from rag.loader import PolicyChunk, load_policy_documents

RRF_K = 60  # standard RRF constant


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    category: str
    heading: str
    rrf_score: float
    chroma_rank: int | None = None
    bm25_rank: int | None = None
    chroma_distance: float | None = None
    bm25_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Combine ChromaDB dense retrieval and BM25 sparse retrieval."""

    def __init__(
        self,
        chunks: list[PolicyChunk] | None = None,
        policies_dir: Path | None = None,
        chroma_persist_dir: Path | None = None,
    ) -> None:
        if chunks is None:
            chunks = load_policy_documents(policies_dir)
        self._chunks = chunks
        self._bm25 = BM25Retriever(chunks)
        self._chroma_persist_dir = chroma_persist_dir

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        chroma_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Query both retrievers, merge with RRF, and return top_k results."""
        k_chroma = chroma_top_k or top_k * 2
        k_bm25 = bm25_top_k or top_k * 2

        chroma_results = query_chroma(
            query_text,
            top_k=k_chroma,
            persist_dir=self._chroma_persist_dir,
        )
        bm25_results = self._bm25.query(query_text, top_k=k_bm25)

        return self._merge_rrf(chroma_results, bm25_results, top_k)

    def _merge_rrf(
        self,
        chroma_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge results using Reciprocal Rank Fusion."""
        candidates: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(chroma_results, start=1):
            cid = hit["chunk_id"]
            if cid not in candidates:
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": hit.get("text", ""),
                    "source": hit.get("source", ""),
                    "category": hit.get("category", ""),
                    "heading": hit.get("heading", ""),
                    "rrf_score": 0.0,
                    "chroma_rank": None,
                    "bm25_rank": None,
                    "chroma_distance": None,
                    "bm25_score": None,
                }
            candidates[cid]["chroma_rank"] = rank
            candidates[cid]["chroma_distance"] = hit.get("distance")
            candidates[cid]["rrf_score"] += 1.0 / (RRF_K + rank)

        for rank, hit in enumerate(bm25_results, start=1):
            cid = hit["chunk_id"]
            if cid not in candidates:
                candidates[cid] = {
                    "chunk_id": cid,
                    "text": hit.get("text", ""),
                    "source": hit.get("source", ""),
                    "category": hit.get("category", ""),
                    "heading": hit.get("heading", ""),
                    "rrf_score": 0.0,
                    "chroma_rank": None,
                    "bm25_rank": None,
                    "chroma_distance": None,
                    "bm25_score": None,
                }
            candidates[cid]["bm25_rank"] = rank
            candidates[cid]["bm25_score"] = hit.get("bm25_score")
            candidates[cid]["rrf_score"] += 1.0 / (RRF_K + rank)

        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: c["rrf_score"],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for c in sorted_candidates[:top_k]:
            results.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    source=c["source"],
                    category=c["category"],
                    heading=c["heading"],
                    rrf_score=c["rrf_score"],
                    chroma_rank=c["chroma_rank"],
                    bm25_rank=c["bm25_rank"],
                    chroma_distance=c["chroma_distance"],
                    bm25_score=c["bm25_score"],
                )
            )

        return results
