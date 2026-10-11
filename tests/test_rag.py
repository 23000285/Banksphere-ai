"""Tests for Phase 8: Policy corpus, ChromaDB ingestion, BM25, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.loader import PolicyChunk, load_policy_documents, POLICIES_DIR
from rag.chroma_store import ingest_chunks, query_chroma, delete_collection
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def policy_chunks() -> list[PolicyChunk]:
    """Load the real policy corpus."""
    return load_policy_documents()


@pytest.fixture()
def chroma_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary directory for ChromaDB."""
    d = tmp_path / "chroma_test"
    d.mkdir()
    return d


@pytest.fixture()
def ingested_chroma(policy_chunks: list[PolicyChunk], chroma_dir: Path) -> Path:
    """Ingest the policy corpus into a temp ChromaDB and return the dir."""
    ingest_chunks(policy_chunks, persist_dir=chroma_dir)
    return chroma_dir


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

class TestPolicyLoading:
    def test_loads_all_policy_files(self, policy_chunks: list[PolicyChunk]) -> None:
        assert len(policy_chunks) > 0
        sources = {c.source for c in policy_chunks}
        assert "kyc_verification.md" in sources
        assert "loan_eligibility.md" in sources
        assert "fraud_disputes.md" in sources
        assert "card_rules.md" in sources
        assert "minor_accounts.md" in sources
        assert "account_closure.md" in sources

    def test_chunk_has_required_fields(self, policy_chunks: list[PolicyChunk]) -> None:
        for chunk in policy_chunks:
            assert chunk.chunk_id
            assert chunk.source
            assert chunk.category
            assert chunk.text
            assert chunk.chunk_index >= 0

    def test_chunk_ids_are_unique(self, policy_chunks: list[PolicyChunk]) -> None:
        ids = [c.chunk_id for c in policy_chunks]
        assert len(ids) == len(set(ids))

    def test_categories_are_set(self, policy_chunks: list[PolicyChunk]) -> None:
        categories = {c.category for c in policy_chunks}
        assert "KYC" in categories
        assert "Loans" in categories
        assert "Fraud" in categories
        assert "Cards" in categories

    def test_metadata_preserved(self, policy_chunks: list[PolicyChunk]) -> None:
        for chunk in policy_chunks:
            assert chunk.metadata["source"] == chunk.source
            assert chunk.metadata["category"] == chunk.category

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_policies"
        empty_dir.mkdir()
        result = load_policy_documents(policies_dir=empty_dir)
        assert result == []

    def test_nonexistent_directory_returns_empty(self, tmp_path: Path) -> None:
        result = load_policy_documents(policies_dir=tmp_path / "does_not_exist")
        assert result == []


# ---------------------------------------------------------------------------
# ChromaDB Ingestion
# ---------------------------------------------------------------------------

class TestChromaIngestion:
    def test_ingest_returns_count(
        self, policy_chunks: list[PolicyChunk], chroma_dir: Path
    ) -> None:
        count = ingest_chunks(policy_chunks, persist_dir=chroma_dir)
        assert count == len(policy_chunks)

    def test_repeat_ingestion_is_idempotent(
        self, policy_chunks: list[PolicyChunk], chroma_dir: Path
    ) -> None:
        ingest_chunks(policy_chunks, persist_dir=chroma_dir)
        ingest_chunks(policy_chunks, persist_dir=chroma_dir)
        results = query_chroma("KYC", top_k=100, persist_dir=chroma_dir)
        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_ingest_empty_chunks(self, chroma_dir: Path) -> None:
        count = ingest_chunks([], persist_dir=chroma_dir)
        assert count == 0

    def test_updated_document_overwrites(
        self, chroma_dir: Path
    ) -> None:
        chunk_v1 = PolicyChunk(
            chunk_id="test::chunk_0",
            source="test.md",
            category="Test",
            heading="Version 1",
            text="Original content about loan eligibility.",
            chunk_index=0,
        )
        chunk_v2 = PolicyChunk(
            chunk_id="test::chunk_0",
            source="test.md",
            category="Test",
            heading="Version 2",
            text="Updated content about card blocking rules.",
            chunk_index=0,
        )
        ingest_chunks([chunk_v1], persist_dir=chroma_dir)
        ingest_chunks([chunk_v2], persist_dir=chroma_dir)

        results = query_chroma("card blocking", top_k=5, persist_dir=chroma_dir)
        matched = [r for r in results if r["chunk_id"] == "test::chunk_0"]
        assert len(matched) == 1
        assert "Updated content" in matched[0]["text"]

    def test_shortened_document_removes_stale_chunks(
        self, chroma_dir: Path
    ) -> None:
        """When a document loses sections, stale chunk IDs are deleted."""
        chunks_v1 = [
            PolicyChunk(
                chunk_id="doc::chunk_0", source="doc.md", category="Test",
                heading="Section 1", text="First section content.", chunk_index=0,
            ),
            PolicyChunk(
                chunk_id="doc::chunk_1", source="doc.md", category="Test",
                heading="Section 2", text="Second section content.", chunk_index=1,
            ),
            PolicyChunk(
                chunk_id="doc::chunk_2", source="doc.md", category="Test",
                heading="Section 3", text="Third section content.", chunk_index=2,
            ),
        ]
        ingest_chunks(chunks_v1, persist_dir=chroma_dir)

        chunks_v2 = [
            PolicyChunk(
                chunk_id="doc::chunk_0", source="doc.md", category="Test",
                heading="Section 1", text="First section revised.", chunk_index=0,
            ),
        ]
        ingest_chunks(chunks_v2, persist_dir=chroma_dir)

        results = query_chroma("section content", top_k=10, persist_dir=chroma_dir)
        ids_found = {r["chunk_id"] for r in results}
        assert "doc::chunk_1" not in ids_found
        assert "doc::chunk_2" not in ids_found


# ---------------------------------------------------------------------------
# ChromaDB Retrieval
# ---------------------------------------------------------------------------

class TestChromaRetrieval:
    def test_kyc_query_returns_kyc_chunks(self, ingested_chroma: Path) -> None:
        results = query_chroma(
            "KYC identity verification documents",
            top_k=5,
            persist_dir=ingested_chroma,
        )
        assert len(results) > 0
        sources = {r.get("source", "") for r in results}
        assert "kyc_verification.md" in sources

    def test_results_have_metadata(self, ingested_chroma: Path) -> None:
        results = query_chroma("loan eligibility", top_k=3, persist_dir=ingested_chroma)
        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "source" in r
            assert "category" in r

    def test_results_have_distance(self, ingested_chroma: Path) -> None:
        results = query_chroma("fraud dispute", top_k=3, persist_dir=ingested_chroma)
        for r in results:
            assert "distance" in r
            assert isinstance(r["distance"], float)

    def test_empty_collection_returns_empty(self, chroma_dir: Path) -> None:
        results = query_chroma("anything", top_k=5, persist_dir=chroma_dir)
        assert results == []

    def test_top_k_limits_results(self, ingested_chroma: Path) -> None:
        results = query_chroma("banking policy", top_k=2, persist_dir=ingested_chroma)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------------------------

class TestBM25Retrieval:
    def test_kyc_query_returns_relevant(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("KYC identity verification", top_k=5)
        assert len(results) > 0
        assert any("kyc_verification.md" == r["source"] for r in results)

    def test_loan_query_returns_relevant(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("personal loan eligibility income", top_k=5)
        assert len(results) > 0
        assert any("loan_eligibility.md" == r["source"] for r in results)

    def test_results_have_metadata(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("card blocking replacement", top_k=3)
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "source" in r
            assert "category" in r
            assert "bm25_score" in r

    def test_top_k_limits_results(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("account", top_k=2)
        assert len(results) <= 2

    def test_empty_corpus(self) -> None:
        retriever = BM25Retriever([])
        assert retriever.corpus_size == 0
        results = retriever.query("anything", top_k=5)
        assert results == []

    def test_no_match_query(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("xyzzyplugh", top_k=5)
        assert results == []

    def test_bm25_scores_are_positive(self, policy_chunks: list[PolicyChunk]) -> None:
        retriever = BM25Retriever(policy_chunks)
        results = retriever.query("fraud transaction dispute", top_k=5)
        for r in results:
            assert r["bm25_score"] > 0


# ---------------------------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def test_hybrid_returns_results(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("KYC verification requirements", top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_hybrid_deduplicates(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("loan eligibility income", top_k=10)
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_hybrid_preserves_metadata(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("card blocking", top_k=5)
        for r in results:
            assert r.chunk_id
            assert r.source
            assert r.category
            assert r.text
            assert r.rrf_score > 0

    def test_hybrid_rrf_scoring(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("fraud dispute transaction", top_k=5)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_top_k(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("banking", top_k=3)
        assert len(results) <= 3

    def test_hybrid_empty_corpus(self, chroma_dir: Path) -> None:
        retriever = HybridRetriever(
            chunks=[],
            chroma_persist_dir=chroma_dir,
        )
        results = retriever.query("anything", top_k=5)
        assert results == []

    def test_hybrid_no_match(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("xyzzyplugh", top_k=5)
        # ChromaDB may return results even for nonsense queries (by distance),
        # but BM25 returns none. If only chroma returns, results may exist.
        # The key invariant is no crash and all results are valid.
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_hybrid_only_bm25_results(
        self, policy_chunks: list[PolicyChunk], chroma_dir: Path
    ) -> None:
        """When chroma is empty but BM25 has data, hybrid still returns results."""
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=chroma_dir,
        )
        results = retriever.query("KYC verification", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.bm25_rank is not None
            assert r.chroma_rank is None

    def test_hybrid_only_chroma_results(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        """When BM25 has no match but chroma does, hybrid still returns results."""
        retriever = HybridRetriever(
            chunks=[],
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("KYC verification", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.chroma_rank is not None
            assert r.bm25_rank is None

    def test_source_filenames_preserved_end_to_end(
        self, policy_chunks: list[PolicyChunk], ingested_chroma: Path
    ) -> None:
        retriever = HybridRetriever(
            chunks=policy_chunks,
            chroma_persist_dir=ingested_chroma,
        )
        results = retriever.query("minor account guardian", top_k=5)
        sources = {r.source for r in results}
        assert any(s.endswith(".md") for s in sources)
