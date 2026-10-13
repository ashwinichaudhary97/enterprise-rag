"""Retrieval evaluation metrics.

Three metrics, computed per question and aggregated across the question set:

    Recall@K   - fraction of relevant docs found in the top-K
    nDCG@K     - rank-weighted relevance, normalized to [0, 1]
    MRR        - mean reciprocal rank of the first relevant result

All functions take lists of doc_ids (strings). Questions with empty
expected_doc_ids (info_not_found category) are skipped, since recall
and MRR are undefined when there is no ground truth.
"""

from __future__ import annotations

import math
from typing import Any


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Remove duplicates while keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
    k: int = 5,
) -> float | None:
    """Fraction of expected docs present in the top-K retrieved.

    Chunks from the same document count once. Returns None if there is
    no ground truth (info_not_found category).
    """
    if not expected_doc_ids:
        return None

    top_k = _dedupe_preserve_order(retrieved_doc_ids[:k])
    expected_set = set(expected_doc_ids)
    hits = sum(1 for d in top_k if d in expected_set)
    return hits / len(expected_set)


def mrr(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> float | None:
    """Reciprocal rank of the first relevant doc. 0 if none found."""
    if not expected_doc_ids:
        return None

    expected_set = set(expected_doc_ids)
    seen: set[str] = set()
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    expected_doc_ids: list[str],
    k: int = 5,
) -> float | None:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance (in expected set = 1, else 0). Normalized by the
    ideal DCG for the given number of expected docs.
    """
    if not expected_doc_ids:
        return None

    expected_set = set(expected_doc_ids)

    # DCG over retrieved (dedupe doc_ids, keep first occurrence)
    dcg = 0.0
    seen: set[str] = set()
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)
        if doc_id in expected_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all expected docs at top positions
    idcg = 0.0
    for rank in range(1, min(len(expected_set), k) + 1):
        idcg += 1.0 / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, float]:
    """Average metrics across queries. Skips None values (no ground truth)."""
    if not per_query:
        return {}

    keys = ["recall_at_5", "ndcg_at_5", "mrr"]
    aggregated: dict[str, float] = {}

    for key in keys:
        values = [q[key] for q in per_query if q.get(key) is not None]
        if values:
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = 0.0

    aggregated["num_questions"] = len(per_query)
    aggregated["num_with_ground_truth"] = sum(
        1 for q in per_query if q.get("recall_at_5") is not None
    )
    return aggregated


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple sanity checks
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    expected = ["doc_b", "doc_e"]

    print("Test 1: two expected, both found in top-5")
    print(f"  recall@5 = {recall_at_k(retrieved, expected, k=5)}")
    print(f"  mrr      = {mrr(retrieved, expected)}")
    print(f"  ndcg@5   = {ndcg_at_k(retrieved, expected, k=5):.4f}")
    print()

    print("Test 2: only one of two expected found")
    expected2 = ["doc_b", "doc_z"]
    print(f"  recall@5 = {recall_at_k(retrieved, expected2, k=5)}")
    print(f"  mrr      = {mrr(retrieved, expected2)}")
    print(f"  ndcg@5   = {ndcg_at_k(retrieved, expected2, k=5):.4f}")
    print()

    print("Test 3: no ground truth (info_not_found)")
    print(f"  recall@5 = {recall_at_k(retrieved, [], k=5)}")
    print(f"  mrr      = {mrr(retrieved, [])}")
    print(f"  ndcg@5   = {ndcg_at_k(retrieved, [], k=5)}")
    print()

    print("Test 4: duplicate doc_ids in retrieved (same doc, multiple chunks)")
    retrieved_dupe = ["doc_a", "doc_a", "doc_b", "doc_a", "doc_c"]
    expected_dupe = ["doc_b"]
    print(f"  recall@5 = {recall_at_k(retrieved_dupe, expected_dupe, k=5)}")
    print(f"  mrr      = {mrr(retrieved_dupe, expected_dupe)}")
    print(f"  ndcg@5   = {ndcg_at_k(retrieved_dupe, expected_dupe, k=5):.4f}")