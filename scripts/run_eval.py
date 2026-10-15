"""Run all four retrieval configs against the 500 benchmark questions.

Produces:
    results/leaderboard.md   - human-readable table
    results/raw_metrics.json - per-question details for later analysis

Configs evaluated:
    dense         - dense retrieval, top-5
    bm25          - BM25 retrieval, top-5
    hybrid        - dense + BM25 RRF fusion, top-5
    hybrid_rerank - hybrid top-20 -> cross-encoder -> top-5

Checkpointing:
    After each config completes, results are saved to
    results/checkpoint_<config>.json. Rerank checkpoints every 25
    questions. On re-run, completed configs are loaded from
    checkpoints and skipped.

Exception logging:
    Every per-question retrieval call is wrapped in try/except.
    If a question crashes, we print the question ID, the exception
    type, and the full traceback, then halt the run.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

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
CHECKPOINT_EVERY = 25
RESULTS_DIR = Path("results")


def save_checkpoint(config_name: str, per_query: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"checkpoint_{config_name}.json"
    path.write_text(
        json.dumps(per_query, indent=2, default=str),
        encoding="utf-8",
    )


def load_checkpoint(config_name: str) -> list[dict[str, Any]] | None:
    path = RESULTS_DIR / f"checkpoint_{config_name}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return None


def load_questions() -> list[dict[str, Any]]:
    ds = load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench",
        "questions",
        split="test",
    )
    return [dict(row) for row in ds]


def _retrieved_doc_ids(results: list[dict[str, Any]]) -> list[str]:
    return [r["doc_id"] for r in results]


def _evaluate_one(
    question: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
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


def _safe_run_one(
    config_name: str,
    i: int,
    q: dict[str, Any],
    runner: Callable[[dict[str, Any]], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Run one question's retrieval with exception logging."""
    try:
        retrieved = runner(q)
        return _evaluate_one(q, retrieved)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"EXCEPTION in config '{config_name}' at question {i}")
        print(f"  question_id: {q.get('question_id', '?')}")
        print(f"  question:    {str(q.get('question', ''))[:120]}")
        print(f"  error type:  {type(e).__name__}")
        print(f"  error msg:   {e}")
        print("=" * 60)
        traceback.print_exc()
        raise


def run_config_dense(questions, store, embedder):
    print("\n=== Config: dense ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        result = _safe_run_one(
            "dense", i, q,
            lambda qq: dense_retrieve(qq["question"], store, embedder, top_k=TOP_K),
        )
        per_query.append(result)
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_bm25(questions, bm25):
    print("\n=== Config: bm25 ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        result = _safe_run_one(
            "bm25", i, q,
            lambda qq: bm25.retrieve(qq["question"], top_k=TOP_K),
        )
        per_query.append(result)
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_hybrid(questions, store, embedder, bm25):
    print("\n=== Config: hybrid ===")
    t0 = time.time()
    per_query = []
    for i, q in enumerate(questions, 1):
        if i % 50 == 0:
            print(f"  {i}/{len(questions)}  ({time.time() - t0:.1f}s)")
        result = _safe_run_one(
            "hybrid", i, q,
            lambda qq: hybrid_retrieve(
                qq["question"], store, embedder, bm25,
                top_k=TOP_K, per_ranker_k=CANDIDATE_POOL,
            ),
        )
        per_query.append(result)
    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def run_config_hybrid_rerank(questions, store, embedder, bm25, reranker):
    print("\n=== Config: hybrid_rerank ===")

    existing = load_checkpoint("hybrid_rerank")
    start_idx = len(existing) if existing else 0
    per_query = list(existing) if existing else []

    if start_idx > 0:
        print(f"  Resuming from checkpoint at question {start_idx}/{len(questions)}")
    else:
        print(f"  Fresh start: {len(questions)} questions x {CANDIDATE_POOL} candidates")

    t0 = time.time()
    for i in range(start_idx, len(questions)):
        q = questions[i]
        result = _safe_run_one(
            "hybrid_rerank", i + 1, q,
            lambda qq: rerank_retrieve(
                qq["question"], store, embedder, bm25, reranker,
                top_k=TOP_K, candidate_pool=CANDIDATE_POOL,
            ),
        )
        per_query.append(result)

        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == len(questions):
            save_checkpoint("hybrid_rerank", per_query)
            elapsed = time.time() - t0
            processed = done - start_idx
            remaining = len(questions) - done
            eta = (elapsed / processed) * remaining if processed > 0 else 0
            print(
                f"  {done}/{len(questions)}  elapsed={elapsed:.1f}s  "
                f"eta={eta:.0f}s  (checkpoint saved)"
            )

    print(f"  done in {time.time() - t0:.1f}s")
    return per_query


def write_leaderboard(configs, output_path: Path) -> None:
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
    rows.sort(key=lambda r: -r["recall_at_5"])

    lines = []
    lines.append("# Retrieval Leaderboard")
    lines.append("")
    lines.append(
        f"Evaluated on {rows[0]['n_ground_truth']} benchmark questions "
        f"with ground truth (info_not_found questions excluded)."
    )
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


def write_raw_metrics(configs, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(configs, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Raw metrics written to {output_path}")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    print("\nSetting up retrieval components...")
    store = VectorStore()
    embedder = Embedder()
    bm25 = BM25Retriever(store)
    reranker = Reranker()

    configs: dict[str, list[dict[str, Any]]] = {}

    def _run_or_load(name: str, runner):
        existing = load_checkpoint(name)
        if existing and len(existing) == len(questions):
            print(f"\n=== Config: {name} (loaded from checkpoint) ===")
            return existing
        result = runner()
        save_checkpoint(name, result)
        return result

    configs["dense"] = _run_or_load(
        "dense", lambda: run_config_dense(questions, store, embedder)
    )
    configs["bm25"] = _run_or_load(
        "bm25", lambda: run_config_bm25(questions, bm25)
    )
    configs["hybrid"] = _run_or_load(
        "hybrid", lambda: run_config_hybrid(questions, store, embedder, bm25)
    )
    configs["hybrid_rerank"] = _run_or_load(
        "hybrid_rerank",
        lambda: run_config_hybrid_rerank(questions, store, embedder, bm25, reranker),
    )

    write_leaderboard(configs, RESULTS_DIR / "leaderboard.md")
    write_raw_metrics(configs, RESULTS_DIR / "raw_metrics.json")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())