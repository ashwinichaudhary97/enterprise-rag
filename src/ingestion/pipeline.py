"""Ingestion pipeline driver.

Runs the full pipeline end-to-end:
    load -> chunk -> embed -> store

Invoke with:
    python -m src.ingestion.pipeline

Reads config from configs/ingestion.yaml. Writes the ChromaDB index to
the persist_directory specified in the config. The pipeline is
idempotent: it resets the collection before adding, so re-running does
not duplicate chunks.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from src.ingestion.chunker import chunk_corpus
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_config, load_slice_cached
from src.ingestion.store import VectorStore


def run_ingestion(config: dict[str, Any], reset: bool = True) -> dict[str, Any]:
    """Run the full ingestion pipeline. Returns a stats dict."""
    stats: dict[str, Any] = {}
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Load documents")
    print("=" * 60)
    t0 = time.time()
    docs = load_slice_cached(config)
    stats["docs_loaded"] = len(docs)
    stats["load_seconds"] = round(time.time() - t0, 2)
    print(f"Loaded {len(docs)} documents in {stats['load_seconds']}s")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 2: Chunk documents")
    print("=" * 60)
    t0 = time.time()
    chunks = chunk_corpus(docs, config)
    stats["chunks_created"] = len(chunks)
    stats["chunk_seconds"] = round(time.time() - t0, 2)
    print(f"Created {len(chunks)} chunks in {stats['chunk_seconds']}s")
    if docs:
        stats["avg_chunks_per_doc"] = round(len(chunks) / len(docs), 2)
        print(f"Average chunks per doc: {stats['avg_chunks_per_doc']}")

    # ------------------------------------------------------------------
    # 3. Embed
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 3: Embed chunks")
    print("=" * 60)
    t0 = time.time()
    emb_cfg = config["embedding"]
    embedder = Embedder(
        model_name=emb_cfg["model_name"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg.get("normalize", True),
    )
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts, show_progress=True)
    stats["embeddings_shape"] = tuple(embeddings.shape)
    stats["embed_seconds"] = round(time.time() - t0, 2)
    print(f"Embedded {len(texts)} chunks in {stats['embed_seconds']}s")
    print(f"Embeddings shape: {stats['embeddings_shape']}")

    # ------------------------------------------------------------------
    # 4. Store
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 4: Store in ChromaDB")
    print("=" * 60)
    t0 = time.time()
    store_cfg = config["vector_store"]
    store = VectorStore(
        persist_directory=store_cfg["persist_directory"],
        collection_name=store_cfg["collection_name"],
    )

    if reset:
        print("Resetting collection (clean rebuild)...")
        store.reset()

    added = store.add_chunks(chunks, embeddings)
    stats["chunks_stored"] = added
    stats["store_seconds"] = round(time.time() - t0, 2)
    stats["collection_total"] = store.count()
    print(f"Stored {added} chunks in {stats['store_seconds']}s")
    print(f"Collection total after store: {stats['collection_total']}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats["total_seconds"] = round(time.time() - t_start, 2)
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Documents loaded:   {stats['docs_loaded']}")
    print(f"Chunks created:     {stats['chunks_created']}")
    print(f"Chunks stored:      {stats['chunks_stored']}")
    print(f"Collection total:   {stats['collection_total']}")
    print(f"Total time:         {stats['total_seconds']}s")
    print()

    return stats


def main() -> int:
    """Entry point. Reads config, runs the pipeline."""
    config_path = "configs/ingestion.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    slice_size = config["slice"]["size"]
    slice_method = config["slice"]["method"]
    model = config["embedding"]["model_name"]
    persist_dir = config["vector_store"]["persist_directory"]

    print(f"Slice size:         {slice_size}")
    print(f"Slice method:       {slice_method}")
    print(f"Embedding model:    {model}")
    print(f"Persist directory:  {persist_dir}")
    print()

    run_ingestion(config, reset=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())