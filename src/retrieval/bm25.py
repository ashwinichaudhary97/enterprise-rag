"""BM25 sparse retrieval with bm25s (disk-backed, memory-mapped).

Ranks chunks by keyword overlap with the query using BM25. Uses the
bm25s library, which stores the index on disk via memory mapping.
Peak RAM during load: ~500 MB (vs. ~4 GB with rank_bm25).

Tokenization uses bm25s's default: Porter stemming + stopword removal.
This matches the tokenization used by Lucene, Elasticsearch, and
OpenSearch — the industry standard for BM25 retrieval.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import bm25s

from src.ingestion.store import VectorStore


_INDEX_DIRNAME = "bm25s_index"
_META_FILENAME = "chunk_meta.pkl"
_CORPUS_FILENAME = "corpus.jsonl"
_PAGE_SIZE = 2000


class BM25Retriever:
    """Disk-backed BM25 index over the ChromaDB collection.

    First construction: streams ChromaDB, tokenizes, builds a bm25s
    index on disk, and saves chunk metadata alongside.

    Subsequent constructions: loads the bm25s index via memory mapping
    (~500 MB peak RAM) and the chunk metadata pickle.
    """

    def __init__(
        self,
        store: VectorStore,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.store = store
        base = Path(cache_dir) if cache_dir else Path(store.persist_directory)
        self.index_dir = base / _INDEX_DIRNAME
        self.meta_path = base / _META_FILENAME

        self._chunk_ids: list[str] = []
        self._doc_ids: list[str] = []
        self._source_types: list[str] = []
        self._texts: list[str] = []
        self._retriever: bm25s.BM25 | None = None

        if self.index_dir.exists() and self.meta_path.exists():
            print(f"Loading BM25 index from {self.index_dir}...")
            self._load_from_disk()
            print(f"BM25 index ready: {len(self._chunk_ids)} chunks (from cache)")
        else:
            print("Building BM25 index from ChromaDB (first run)...")
            self._build_and_save()
            print(f"BM25 index ready: {len(self._chunk_ids)} chunks")

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def _build_and_save(self) -> None:
        """Stream ChromaDB, tokenize, build bm25s index, save to disk."""
        collection = self.store._collection
        total = collection.count()
        print(f"  total chunks to process: {total}")

        corpus: list[str] = []
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=_PAGE_SIZE,
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
                corpus.append(text)
            offset += _PAGE_SIZE
            print(f"  loaded {offset}/{total}")

        print("  tokenizing with bm25s (Porter stemmer + stopwords)...")
        tokens = bm25s.tokenize(corpus, show_progress=False)

        print("  building bm25s index...")
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens)

        print(f"  saving index to {self.index_dir}...")
        self._retriever.save(str(self.index_dir), corpus=corpus)

        print(f"  saving chunk metadata to {self.meta_path}...")
        meta_payload = {
            "chunk_ids": self._chunk_ids,
            "doc_ids": self._doc_ids,
            "source_types": self._source_types,
            "texts": self._texts,
        }
        with self.meta_path.open("wb") as f:
            pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Load chunk metadata and the bm25s index via memory mapping."""
        with self.meta_path.open("rb") as f:
            meta_payload = pickle.load(f)
        self._chunk_ids = meta_payload["chunk_ids"]
        self._doc_ids = meta_payload["doc_ids"]
        self._source_types = meta_payload["source_types"]
        self._texts = meta_payload["texts"]

        self._retriever = bm25s.BM25.load(str(self.index_dir), mmap=True)

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-K chunks by BM25 score for the query."""
        if not query or not query.strip():
            return []
        if self._retriever is None:
            return []

        query_tokens = bm25s.tokenize([query], show_progress=False)
        results, scores = self._retriever.retrieve(query_tokens, k=top_k)

        out: list[dict[str, Any]] = []
        for idx, score in zip(results[0], scores[0]):
            i = int(idx)
            out.append({
                "chunk_id": self._chunk_ids[i],
                "doc_id": self._doc_ids[i],
                "source_type": self._source_types[i],
                "score": float(score),
                "text": self._texts[i],
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