"""Run all four retrieval configs against the 500 benchmark questions.

Produces:
    results/leaderboard.md   - human-readable table
    results/raw_metrics.json - per-question details for later analysis

Configs evaluated:
    dense         - dense retrieval, top-5
    bm25          - BM25 retrieval, top-5
    hybrid        - dense + BM25 RRF fusion, top-5
    hybrid_rerank - hybrid top-20 -> cross-encoder -> top-5

Note: questions with empty expected_doc_ids (info_not_found) are
skipped by retrieval metrics, since recall/MRR/nDCG are undefined
without ground truth.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.evaluation.metrics import aggregate_metrics, mrr, ndcg_at_k, recall_at_k
from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import dense_retrieve
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.rerank import Reranker, rerank_retrieve


TOP_K = 5
CANDIDATE_POOL = 20


def load_questions() -> list[dict[str, Any]]:
    """Load the 500 benchmark questions from HF cache."""
    ds = load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench",
        "questions",
        split="test",
    )
    return [dict(row) for row in ds]


def _retrieved_doc_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract doc_ids from a list of retrieval results, in order."""
    return [r["doc_id"] for r in results]


def _evaluate_one(
    question: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute metrics for one question given retrieval results."""
    expected = question.get("expected_doc_ids", []) or []
    doc_ids = _retrieved_doc_ids(retrieved)
    return {
        "question_id": question["question_id"],
        "question_type": question.get("question_type", ""),
        "num_expected": len(expected),
        "recall_at_5": recall_at_k(doc_ids, expected, k=TOP_K),
        "mrr": mrr(doc_ids, expected),
        "ndcg_at_5": ndcg_at_k(doc_ids, expected, k=TOP_K),
    }


def run_config_dense(questions, store, embedder):
    """Run dense config on all questions."""
    print("\n=== Config: dense ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        retrieved = dense_retrieve(q["question"], store, embedder, top_k=TOP_K)
        per_query.append(_evaluate_one(q, retrieved))
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_bm25(questions, bm25):
    """Run BM25 config on all questions."""
    print("\n=== Config: bm25 ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        retrieved = bm25.retrieve(q["question"], top_k=TOP_K)
        per_query.append(_evaluate_one(q, retrieved))
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_hybrid(questions, store, embedder, bm25):
    """Run hybrid RRF config on all questions."""
    print("\n=== Config: hybrid ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        retrieved = hybrid_retrieve(
            q["question"], store, embedder, bm25,
            top_k=TOP_K, per_ranker_k=CANDIDATE_POOL,
        )
        per_query.append(_evaluate_one(q, retrieved))
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_hybrid_rerank(questions, store, embedder, bm25, reranker):
    """Run hybrid + rerank config on all questions."""
    print("\n=== Config: hybrid_rerank ===")
    print(f"  (slowest config: {len(questions)} questions x {CANDIDATE_POOL} candidates)")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 25 == 0 or i == len(questions):
            elapsed = time.time() - t0
            eta = (elapsed / i) * (len(questions) - i) if i > 0 else 0
            print(f"  {i}/{len(questions)}  elapsed={elapsed:.1f}s  eta={eta:.0f}s")
        retrieved = rerank_retrieve(
            q["question"], store, embedder, bm25, reranker,
            top_k=TOP_K, candidate_pool=CANDIDATE_POOL,
        )
        per_query.append(_evaluate_one(q, retrieved))
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def write_leaderboard(
    configs: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Write the leaderboard markdown file."""
    rows = []
    for name, per_query in configs.items():
        agg = aggregate_metrics(per_query)
        rows.append({
            "config": name,
            "recall_at_5": agg.get("recall_at_5", 0.0),
            "ndcg_at_5": agg.get("ndcg_at_5", 0.0),
            "mrr": agg.get("mrr", 0.0),
            "n_ground_truth": agg.get("num_with_ground_truth", 0),
        })

    # Sort by recall@5 descending
    rows.sort(key=lambda r: -r["recall_at_5"])

    lines = []
    lines.append("# Retrieval Leaderboard")
    lines.append("")
    lines.append(f"Evaluated on {rows[0]['n_ground_truth']} benchmark questions "
                 f"with ground truth (info_not_found questions excluded).")
    lines.append("")
    lines.append("| Rank | Config | Recall@5 | nDCG@5 | MRR |")
    lines.append("|------|--------|----------|--------|-----|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['config']} | {r['recall_at_5']:.4f} | "
            f"{r['ndcg_at_5']:.4f} | {r['mrr']:.4f} |"
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLeaderboard written to {output_path}")


def write_raw_metrics(
    configs: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Write per-question metrics as JSON for later analysis."""
    output_path.write_text(
        json.dumps(configs, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Raw metrics written to {output_path}")


def main() -> int:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    print("\nSetting up retrieval components...")
    store = VectorStore()
    embedder = Embedder()
    bm25 = BM25Retriever(store)
    reranker = Reranker()

    configs: dict[str, list[dict[str, Any]]] = {}

    configs["dense"] = run_config_dense(questions, store, embedder)
    configs["bm25"] = run_config_bm25(questions, bm25)
    configs["hybrid"] = run_config_hybrid(questions, store, embedder, bm25)
    configs["hybrid_rerank"] = run_config_hybrid_rerank(
        questions, store, embedder, bm25, reranker
    )

    write_leaderboard(configs, results_dir / "leaderboard.md")
    write_raw_metrics(configs, results_dir / "raw_metrics.json")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())