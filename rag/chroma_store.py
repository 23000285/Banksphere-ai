"""ChromaDB vector store for policy document retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from rag.loader import PolicyChunk

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "banking_policies"


def _get_client(persist_dir: Path | None = None) -> chromadb.ClientAPI:
    directory = persist_dir or DEFAULT_PERSIST_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(directory),
        settings=Settings(anonymized_telemetry=False),
    )


def ingest_chunks(
    chunks: list[PolicyChunk],
    persist_dir: Path | None = None,
) -> int:
    """Upsert policy chunks into ChromaDB, returning the count ingested.

    Uses chunk_id as the document ID so repeat ingestion is idempotent —
    updated documents overwrite their previous embeddings.  Stale chunks
    (IDs present in the collection but absent from the new batch) are
    deleted so that shortened or removed documents don't leave orphans.
    """
    client = _get_client(persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    new_ids = {c.chunk_id for c in chunks}

    existing = collection.get()
    stale_ids = [eid for eid in existing["ids"] if eid not in new_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    if not chunks:
        return 0

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas: list[dict[str, Any]] = [
        {
            "source": c.source,
            "category": c.category,
            "heading": c.heading,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def query_chroma(
    query: str,
    top_k: int = 5,
    persist_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Query ChromaDB and return ranked results with metadata and distance."""
    client = _get_client(persist_dir)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    output: list[dict[str, Any]] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return output

    for i, doc_id in enumerate(results["ids"][0]):
        entry: dict[str, Any] = {
            "chunk_id": doc_id,
            "text": results["documents"][0][i] if results["documents"] else "",
            "distance": results["distances"][0][i] if results["distances"] else None,
        }
        if results["metadatas"] and results["metadatas"][0]:
            entry.update(results["metadatas"][0][i])
        output.append(entry)

    return output


def delete_collection(persist_dir: Path | None = None) -> None:
    """Delete the policy collection (useful for testing)."""
    client = _get_client(persist_dir)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
