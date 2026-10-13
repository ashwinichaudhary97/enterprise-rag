"""Cross-encoder reranking.

Takes the top-N candidates from hybrid retrieval and re-scores them
with a cross-encoder model that reads query and document together.

The bi-encoder (MiniLM) embeds query and document separately. It is
fast but cannot model token-level interactions. The cross-encoder
takes (query, document) as a single input, produces a relevance score,
and captures interactions that the bi-encoder misses.

The cost: cross-encoders cannot pre-compute document embeddings, so
every (query, doc) pair requires a full model forward pass.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import hybrid_retrieve


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker with lazy model loading."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        print(f"Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)
        print("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Score each candidate with the cross-encoder, return top-K.

        Args:
            query: the user's question
            candidates: list of dicts from hybrid retrieval, each with 'text'
            top_k: how many to keep

        Returns:
            Reranked candidates, sorted by cross-encoder score descending.
            Each item retains the original fields, plus 'ce_score'.
        """
        if not query or not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: -float(x[1]),
        )[:top_k]

        out: list[dict[str, Any]] = []
        for candidate, score in ranked:
            entry = dict(candidate)
            entry["ce_score"] = float(score)
            out.append(entry)
        return out


def rerank_retrieve(
    query: str,
    store: VectorStore,
    embedder: Embedder,
    bm25_retriever: BM25Retriever,
    reranker: Reranker,
    top_k: int = 5,
    candidate_pool: int = 20,
) -> list[dict[str, Any]]:
    """Full hybrid + rerank pipeline for one query.

    1. Hybrid retrieval gets top `candidate_pool` candidates.
    2. Cross-encoder reranks them.
    3. Return top `top_k`.
    """
    candidates = hybrid_retrieve(
        query, store, embedder, bm25_retriever,
        top_k=candidate_pool, per_ranker_k=candidate_pool,
    )
    return reranker.rerank(query, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore()
    embedder = Embedder()
    bm25_retriever = BM25Retriever(store)
    reranker = Reranker()

    query = "deployment issue"
    print()
    print(f"Query: {query!r}")
    print()

    results = rerank_retrieve(
        query, store, embedder, bm25_retriever, reranker,
        top_k=5, candidate_pool=20,
    )

    print(f"Top-{len(results)} after reranking:")
    for i, r in enumerate(results):
        preview = r["text"][:80].replace("\n", " ")
        ranks_str = ", ".join(f"r{k}={v}" for k, v in sorted(r.get("ranks", {}).items()))
        print(
            f"  {i + 1}. ce={r['ce_score']:+.3f}  "
            f"rrf={r['score']:.5f}  source={r['source_type']}  [{ranks_str}]"
        )
        print(f"      {preview}...")