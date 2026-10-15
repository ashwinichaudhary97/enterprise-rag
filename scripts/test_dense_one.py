"""Diagnostic: run one dense retrieval against the 512K index.

If this fails, we see the exact error. If it succeeds, the eval
loop has a bug rather than the retrieval stack.
"""

from __future__ import annotations

import traceback

from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStore
from src.retrieval.dense import dense_retrieve


def main() -> None:
    print("Opening 512K store...")
    store = VectorStore()
    print(f"  collection count: {store.count()}")

    print("Loading embedder...")
    embedder = Embedder()

    print("Running dense retrieval on one query...")
    try:
        results = dense_retrieve(
            "what is the deployment process",
            store,
            embedder,
            top_k=3,
        )
    except Exception:
        print("EXCEPTION during dense_retrieve:")
        traceback.print_exc()
        return

    print(f"Results: {len(results)}")
    for i, r in enumerate(results):
        print(f"  {i + 1}. doc_id={r['doc_id'][:30]}  source={r['source_type']}  score={r['score']:.4f}")


if __name__ == "__main__":
    main()