"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Combines dense and BM25 result lists into a single ranked list. RRF
works on ranks, not scores, so it is invariant to the different score
scales of the two rankers (dense is [0, 1], BM25 is unbounded).

Formula:
    RRF_score(doc) = sum over rankers r of 1 / (k + rank_r(doc))

Standard k = 60 (Cormack et al. 2009).
"""

from __future__ import annotations

from typing import Any

from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import dense_retrieve


DEFAULT_K = 60


def rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    k: int = DEFAULT_K,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists with RRF. Returns top-K fused results.

    Args:
        rankings: list of ranked lists. Each inner list is a list of
                  dicts with at least 'chunk_id'. Sorted best-first.
        k: RRF smoothing constant. Default 60.
        top_k: how many fused results to return.

    Returns:
        Fused top-K. Each dict has chunk_id, doc_id, source_type, text,
        score (RRF), and 'ranks' (dict of ranker_index -> rank).
    """
    scores: dict[str, float] = {}
    payload: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[int, int]] = {}

    for ranker_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in payload:
                # Keep the first-seen payload (metadata is identical across rankers)
                payload[cid] = item
            ranks.setdefault(cid, {})[ranker_idx] = rank

    # Sort by RRF score descending
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

    out: list[dict[str, Any]] = []
    for cid, score in ordered:
        entry = dict(payload[cid])
        entry["score"] = float(score)
        entry["ranks"] = ranks[cid]
        out.append(entry)
    return out


def hybrid_retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    bm25_retriever: BM25Retriever,
    top_k: int = 5,
    per_ranker_k: int = 20,
    k: int = DEFAULT_K,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense + BM25, fused with RRF.

    Args:
        query: the user's question
        store: VectorStore instance
        embedder: Embedder instance
        bm25_retriever: BM25Retriever instance (already built)
        top_k: how many results to return
        per_ranker_k: how many candidates each ranker contributes
        k: RRF smoothing constant

    Returns:
        Top-K fused results.
    """
    dense_results = dense_retrieve(query, store, embedder, top_k=per_ranker_k)
    bm25_results = bm25_retriever.retrieve(query, top_k=per_ranker_k)
    return rrf_fuse([dense_results, bm25_results], k=k, top_k=top_k)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()
    embedder = Embedder()
    bm25_retriever = BM25Retriever(store)

    query = "deployment issue"
    print()
    print(f"Query: {query!r}")
    print()

    results = hybrid_retrieve(query, store, embedder, bm25_retriever, top_k=5)

    print(f"Top-{len(results)} hybrid results:")
    for i, r in enumerate(results):
        preview = r["text"][:80].replace("\n", " ")
        ranks_str = ", ".join(f"r{k}={v}" for k, v in sorted(r["ranks"].items()))
        print(f"  {i + 1}. rrf={r['score']:.5f}  source={r['source_type']}  [{ranks_str}]")
        print(f"      {preview}...")