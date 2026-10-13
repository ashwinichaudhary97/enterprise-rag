"""ChromaDB storage layer for the ingestion pipeline.

Wraps a persistent ChromaDB collection. Handles inserting chunks with
their embeddings and metadata, and provides query and inspection methods.

The same class is used during ingestion (add) and later during retrieval
(query). Metadata filtering (`where` clause) is preserved because access
control in Phase 3 filters at the collection level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_directory: str | Path = "data/indexes/chroma",
        collection_name: str = "enterprise_rag",
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name

        # Ensure the persist directory exists.
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # ChromaDB PersistentClient writes to disk and reloads on next init.
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]] | Any,
        batch_size: int = 500,
    ) -> int:
        """Insert chunks with their embeddings. Returns the count inserted.

        Args:
            chunks: list of dicts with keys: chunk_id, doc_id, source_type,
                    chunk_index, text, metadata
            embeddings: (N, D) array-like, matching chunks order
            batch_size: how many to write per ChromaDB call

        Returns:
            Number of chunks added.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        total = len(chunks)
        added = 0

        # ChromaDB expects plain lists for embeddings, dicts for metadata.
        embeddings_list = [
            emb.tolist() if hasattr(emb, "tolist") else list(emb)
            for emb in embeddings
        ]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]

            self._collection.add(
                ids=[c["chunk_id"] for c in batch_chunks],
                documents=[c["text"] for c in batch_chunks],
                embeddings=embeddings_list[start:end],
                metadatas=[self._sanitize_metadata(c) for c in batch_chunks],
            )
            added += len(batch_chunks)

        return added

    # -----------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float] | Any,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the top-K nearest chunks for a query embedding.

        Args:
            query_embedding: (D,) array-like
            top_k: how many results to return
            where: optional ChromaDB metadata filter (used for RBAC)

        Returns:
            dict with keys: ids, documents, metadatas, distances
        """
        emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)

        kwargs: dict[str, Any] = {
            "query_embeddings": [emb],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self._collection.query(**kwargs)

    def count(self) -> int:
        """Number of chunks currently in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it. Destructive."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB only accepts str, int, float, bool in metadata.

        Flatten the chunk into a single dict with prefixed keys.
        """
        meta = chunk.get("metadata", {}) or {}
        out: dict[str, Any] = {
            "doc_id": str(chunk["doc_id"]),
            "source_type": str(chunk["source_type"]),
            "chunk_index": int(chunk["chunk_index"]),
        }
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            else:
                out[key] = str(value)
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    import numpy as np

    # Use a temporary directory so we do not touch the real index.
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(persist_directory=tmp, collection_name="smoke_test")
        print(f"Opened collection: {store.collection_name}")
        print(f"Initial count: {store.count()}")

        # Fake chunks + fake embeddings
        fake_chunks = [
            {
                "chunk_id": f"c_{i}",
                "doc_id": f"doc_{i}",
                "source_type": "slack",
                "chunk_index": i,
                "text": f"Sample text number {i}",
                "metadata": {"title": f"Title {i}", "token_count": 10},
            }
            for i in range(5)
        ]
        fake_embeddings = np.random.rand(5, 384).astype(np.float32)

        added = store.add_chunks(fake_chunks, fake_embeddings)
        print(f"Added {added} chunks")
        print(f"Count after add: {store.count()}")

        # Query with one of the vectors — should retrieve it as top-1
        result = store.query(fake_embeddings[2], top_k=3)
        print("Query returned:")
        for i, (cid, dist) in enumerate(zip(result["ids"][0], result["distances"][0])):
            print(f"  rank {i + 1}: {cid}  distance={dist:.4f}")

        # Reset and verify empty
        store.reset()
        print(f"Count after reset: {store.count()}")
        print("Smoke test passed.")