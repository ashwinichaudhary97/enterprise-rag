"""Document loader for the ingestion pipeline.

Reads the EnterpriseRAG-Bench documents dataset from the HuggingFace cache,
applies a slice (100 / 1K / 10K), and optionally stratifies by source type.

The output is a plain Python list of dicts, one per document, ready to be
passed to the chunker.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset


def load_config(config_path: str | Path = "configs/ingestion.yaml") -> dict[str, Any]:
    """Read the ingestion YAML config and return it as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_documents(config: dict[str, Any]):
    """Load the full documents dataset from the HF cache (no slicing yet)."""
    dataset_cfg = config["dataset"]
    return load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config_documents"],
        split=dataset_cfg["split"],
    )


def _stratified_sample(
    docs: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample `size` documents, proportional to source_type distribution."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_source[doc["source_type"]].append(doc)

    total = len(docs)
    rng = random.Random(seed)

    sampled: list[dict[str, Any]] = []
    for source, group in by_source.items():
        share = len(group) / total
        n = max(1, round(share * size))
        rng.shuffle(group)
        sampled.extend(group[:n])

    # Trim or pad to hit exact size
    rng.shuffle(sampled)
    if len(sampled) > size:
        sampled = sampled[:size]
    return sampled


def _first_n(docs: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Take the first N documents in dataset order."""
    return docs[:size]


def _random_n(
    docs: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Random sample without stratification."""
    rng = random.Random(seed)
    rng.shuffle(docs)
    return docs[:size]


def load_slice(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load documents, apply the configured slice, return a list of dicts.

    The returned list has one entry per document with keys:
        doc_id, source_type, title, content
    """
    slice_cfg = config["slice"]
    size = int(slice_cfg["size"])
    method = slice_cfg["method"]
    seed = int(slice_cfg.get("random_seed", 42))

    dataset = load_all_documents(config)
    docs = [dict(row) for row in dataset]

    if method == "first":
        return _first_n(docs, size)
    if method == "random":
        return _random_n(docs, size, seed)
    if method == "stratified":
        return _stratified_sample(docs, size, seed)

    raise ValueError(f"Unknown slice method: {method!r}")


if __name__ == "__main__":
    # Quick smoke test: load the slice, print source distribution.
    cfg = load_config()
    docs = load_slice(cfg)

    print(f"Loaded {len(docs)} documents (slice method: {cfg['slice']['method']})")
    counts: dict[str, int] = defaultdict(int)
    for doc in docs:
        counts[doc["source_type"]] += 1

    print("Source distribution:")
    for source, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {n}")