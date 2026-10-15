"""Test bm25s on a sample of the 512K ChromaDB index.

Loads 1,000 chunks from the configured ChromaDB, tokenizes them with
bm25s, builds a bm25s index, and queries for "deployment issue".

This is a small-scale correctness test before running bm25s on the
full 642K-chunk corpus. The sample is large enough that bm25s's
vocabulary pruning does not strip meaningful terms.

Run with:
    python -m scripts.test_bm25s
"""

from __future__ import annotations

import os

import bm25s

from src.ingestion.store import VectorStore


SAMPLE_SIZE = 1000


def main() -> None:
    persist = os.environ.get("CHROMA_PATH", "data/indexes/chroma")
    print(f"CHROMA_PATH: {persist}")

    store = VectorStore(persist_directory=persist)
    total = store.count()
    print(f"Total chunks in collection: {total}")

    print(f"Reading {SAMPLE_SIZE} chunks from ChromaDB...")
    collection = store._collection
    batch = collection.get(
        limit=SAMPLE_SIZE,
        offset=0,
        include=["documents", "metadatas"],
    )
    chunk_ids = batch["ids"]
    texts = batch["documents"]
    print(f"  loaded {len(texts)} chunks")

    print("Tokenizing with bm25s...")
    tokens = bm25s.tokenize(texts, show_progress=False)
    print(f"  tokenized {len(tokens)} chunks")
    print(f"  sample tokens (chunk 0): {list(tokens[0])[:15]}")
    print(f"  sample tokens (chunk 1): {list(tokens[1])[:15]}")

    print("Building bm25s index...")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    print("  index built")

    queries = [
        "deployment issue",
        "rollback procedure",
        "pricing tier enterprise",
    ]

    for query in queries:
        print()
        print(f"Query: {query!r}")
        q_tokens = bm25s.tokenize([query], show_progress=False)
        print(f"  query tokens: {list(q_tokens[0])}")

        results, scores = retriever.retrieve(q_tokens, k=5)
        print("  top-5:")
        for i in range(len(results[0])):
            idx = results[0][i]
            score = scores[0][i]
            preview = texts[idx][:80].replace("\n", " ")
            source = batch["metadatas"][idx].get("source_type", "")
            print(f"    {i + 1}. score={score:.4f}  source={source}")
            print(f"       {preview}...")


if __name__ == "__main__":
    main()