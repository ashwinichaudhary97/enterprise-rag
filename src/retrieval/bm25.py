"""BM25 sparse retrieval.

Ranks chunks by keyword overlap with the query, using the BM25Okapi
algorithm from the rank_bm25 library.

BM25 complements dense retrieval:
    - Dense finds chunks that MEAN the same thing (semantic)
    - BM25 finds chunks that USE the same words (lexical)

Neither wins everywhere. The leaderboard will show where each fails.

The corpus is loaded once from ChromaDB, tokenized, and indexed in
memory. At 12,580 chunks, this takes a few seconds. For larger corpora,
the tokenized corpus would be persisted to disk.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.ingestion.store import VectorStore


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens.

    Drops punctuation, emojis, and whitespace. Keeps digits (version
    numbers, ticket IDs, etc. matter in enterprise data).
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 index over the ChromaDB collection."""

    def __init__(self, store: VectorStore) -> None:
        """Build the BM25 index by reading every chunk from ChromaDB."""
        self.store = store
        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._source_types: list[str] = []
        self._texts: list[str] = []
        self._tokenized: list[list[str]] = []

        print("Building BM25 index from ChromaDB...")
        self._load_corpus()
        self._bm25 = BM25Okapi(self._tokenized)
        print(f"BM25 index ready: {len(self._chunk_ids)} chunks")

    def _load_corpus(self) -> None:
        """Fetch every chunk from ChromaDB and tokenize."""
        collection = self.store._collection  # internal access, documented
        total = collection.count()

        # ChromaDB allows paginated get()
        page_size = 2000
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=page_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for chunk_id, text, meta in zip(
                batch["ids"], batch["documents"], batch["metadatas"]
            ):
                self._chunk_ids.append(chunk_id)
                self._doc_ids.append(meta.get("doc_id", ""))
                self._source_types.append(meta.get("source_type", ""))
                self._texts.append(text)
                self._tokenized.append(tokenize(text))
            offset += page_size
            print(f"  loaded {offset}/{total}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-K chunks by BM25 score for the query."""
        if not query or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-K indices by score (descending)
        top_indices = scores.argsort()[::-1][:top_k]

        out: list[dict[str, Any]] = []
        for idx in top_indices:
            out.append({
                "chunk_id": self._chunk_ids[idx],
                "doc_id": self._doc_ids[idx],
                "source_type": self._source_types[idx],
                "score": float(scores[idx]),
                "text": self._texts[idx],
            })
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()
    retriever = BM25Retriever(store)

    query = "deployment issue"
    print()
    print(f"Query: {query!r}")
    print()

    results = retriever.retrieve(query, top_k=5)

    print(f"Top-{len(results)} results:")
    for i, r in enumerate(results):
        preview = r["text"][:80].replace("\n", " ")
        print(f"  {i + 1}. score={r['score']:.4f}  source={r['source_type']}  doc={r['doc_id'][:20]}")
        print(f"      {preview}...")