"""BM25 keyword retrieval over the policy corpus."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from rag.loader import PolicyChunk


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """BM25 sparse retrieval over a list of PolicyChunks."""

    def __init__(self, chunks: list[PolicyChunk]) -> None:
        self._chunks = list(chunks)
        if self._chunks:
            tokenized = [_tokenize(c.text) for c in self._chunks]
            self._index = BM25Okapi(tokenized)
        else:
            self._index = None

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k BM25-ranked results with text and metadata."""
        if not self._chunks or self._index is None:
            return []

        tokens = _tokenize(query_text)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        scored_chunks = [
            (score, chunk)
            for score, chunk in zip(scores, self._chunks)
            if score > 0
        ]
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, chunk in scored_chunks[:top_k]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "category": chunk.category,
                    "heading": chunk.heading,
                    "bm25_score": float(score),
                }
            )

        return results
