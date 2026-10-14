"""Verify the 512K index downloaded from HuggingFace.

Opens the ChromaDB collection at data/indexes/chroma_512k,
prints the total chunk count, and runs one test query.

If the count is ~642,000 and the query returns sensible results,
the index is confirmed working.
"""

from __future__ import annotations

import os
from pathlib import Path

# Point at the downloaded 512K index
os.environ["CHROMA_PATH"] = str(Path("data/indexes/chroma_512k").resolve())

from src.ingestion.store import VectorStore
from src.ingestion.embedder import Embedder
from src.retrieval.dense import dense_retrieve


def main() -> None:
    print("Opening 512K index...")
    store = VectorStore()
    count = store.count()
    print(f"Collection count: {count}")
    print()

    if count == 0:
        print("ERROR: index is empty. Download may be incomplete.")
        return

    print("Loading embedder...")
    embedder = Embedder()
    print()

    query = "deployment issue"
    print(f"Query: {query!r}")
    print()

    results = dense_retrieve(query, store, embedder, top_k=3)

    print(f"Top-{len(results)} results:")
    for i, r in enumerate(results):
        preview = r["text"][:100].replace("\n", " ")
        print(f"  {i + 1}. score={r['score']:.4f}  source={r['source_type']}")
        print(f"      {preview}...")
    print()

    print("If count is around 642,000 and results look relevant,")
    print("the index is validated and ready for eval.")


if __name__ == "__main__":
    main()