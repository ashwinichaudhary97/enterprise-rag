"""Dense retrieval: embed the query, find top-K nearest chunks.

This is the baseline retrieval method. Every other retrieval approach
(BM25, hybrid, reranked) is compared against it.

The function calls ChromaDB's built-in query method, which uses the
cosine similarity metric configured at collection creation time.
"""

from __future__ import annotations

from typing import Any

from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore


def dense_retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-K chunks for a single query via dense similarity.

    Args:
        query: the user's question
        store: VectorStore instance (ChromaDB collection)
        embedder: Embedder instance (MiniLM)
        top_k: how many results to return
        where: optional ChromaDB metadata filter (used in Phase 3)

    Returns:
        List of dicts, each with:
            chunk_id, doc_id, source_type, score, text
        Sorted by score descending (higher = more similar).
    """
    if not query or not query.strip():
        return []

    query_embedding = embedder.embed_query(query)

    result = store.query(query_embedding, top_k=top_k, where=where)

    # ChromaDB returns parallel lists. Zip them into per-chunk dicts.
    out: list[dict[str, Any]] = []
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite.
        # Convert to similarity in [0, 1] for interpretability.
        score = 1.0 - (distance / 2.0)
        out.append({
            "chunk_id": chunk_id,
            "doc_id": meta.get("doc_id", ""),
            "source_type": meta.get("source_type", ""),
            "score": float(score),
            "distance": float(distance),
            "text": text,
        })

    return out


def dense_retrieve_batch(
    queries: list[str],
    store: VectorStore,
    embedder: Embedder,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    """Retrieve for a list of queries.

    Embeds all queries in a single batch (faster than one at a time),
    then queries ChromaDB per query.

    Returns a list of results, one per query.
    """
    if not queries:
        return []

    # Batch-embed all queries at once (much faster than per-query).
    query_embeddings = embedder.embed_texts(queries, show_progress=True)

    results: list[list[dict[str, Any]]] = []
    for emb in query_embeddings:
        result = store.query(emb, top_k=top_k, where=where)

        out: list[dict[str, Any]] = []
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - (distance / 2.0)
            out.append({
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "source_type": meta.get("source_type", ""),
                "score": float(score),
                "distance": float(distance),
                "text": text,
            })
        results.append(out)

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.loader import load_config

    cfg = load_config()
    store = VectorStore()
    embedder = Embedder()

    query = "deployment issue"
    print(f"Query: {query!r}")
    print(f"Collection count: {store.count()}")
    print()

    results = dense_retrieve(query, store, embedder, top_k=5)

    print(f"Top-{len(results)} results:")
    for i, r in enumerate(results):
        preview = r["text"][:80].replace("\n", " ")
        print(f"  {i + 1}. score={r['score']:.4f}  source={r['source_type']}  doc={r['doc_id'][:20]}")
        print(f"      {preview}...")