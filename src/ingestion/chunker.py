"""Source-aware chunker for the ingestion pipeline.

Splits documents into chunks according to rules per source type:
    - Slack threads stay together
    - Gmail messages are atomic
    - Jira issues are atomic
    - Linear tickets are atomic
    - Confluence splits at markdown headings
    - Google Drive uses recursive character split with overlap
    - HubSpot records are atomic
    - Fireflies splits at speaker turns
    - GitHub PRs/comments are atomic
    - Fallback: recursive character split

Each chunk is a dict with:
    chunk_id, doc_id, source_type, chunk_index, text, metadata
"""

from __future__ import annotations

import hashlib
from typing import Any

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Approximate token count: 1 token ~= 4 characters of English."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Per-source chunkers
# ---------------------------------------------------------------------------

def _make_chunk(
    doc: dict[str, Any],
    text: str,
    index: int,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one chunk dict from a document and a text segment."""
    base_id = f"{doc['doc_id']}_c{index:04d}"
    short_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    metadata = {
        "title": doc.get("title", ""),
        "token_count": estimate_tokens(text),
    }
    if extra_meta:
        metadata.update(extra_meta)
    return {
        "chunk_id": f"{base_id}_{short_hash}",
        "doc_id": doc["doc_id"],
        "source_type": doc["source_type"],
        "chunk_index": index,
        "text": text,
        "metadata": metadata,
    }


def chunk_slack(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Slack: keep thread together, cap at max_tokens."""
    max_tokens = cfg["max_tokens"]
    text = doc["content"]
    if estimate_tokens(text) <= max_tokens:
        return [_make_chunk(doc, text, 0, {"chunking_rule": "slack_thread"})]
    # Fall back to recursive split if the thread is unusually long
    return chunk_recursive(doc, {"chunk_size": max_tokens, "chunk_overlap": 100},
                           rule_name="slack_thread_overflow")


def chunk_gmail(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Gmail: one message per chunk (atomic)."""
    return [_make_chunk(doc, doc["content"], 0, {"chunking_rule": "gmail_message"})]


def chunk_jira(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Jira: one issue per chunk (atomic)."""
    return [_make_chunk(doc, doc["content"], 0, {"chunking_rule": "jira_issue"})]


def chunk_linear(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Linear: one ticket per chunk (atomic)."""
    return [_make_chunk(doc, doc["content"], 0, {"chunking_rule": "linear_ticket"})]


def chunk_hubspot(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """HubSpot: one record per chunk (atomic)."""
    return [_make_chunk(doc, doc["content"], 0, {"chunking_rule": "hubspot_record"})]


def chunk_github(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """GitHub: one PR or comment thread per chunk (atomic)."""
    return [_make_chunk(doc, doc["content"], 0, {"chunking_rule": "github_pr"})]


def chunk_confluence(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Confluence: split at markdown headings (## or #).

    Each heading starts a new chunk. Content before the first heading is
    kept as its own chunk if non-empty.
    """
    max_tokens = cfg.get("max_tokens", 1024)
    text = doc["content"]
    lines = text.split("\n")

    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        is_heading = line.lstrip().startswith("#")
        if is_heading and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    sections = [s for s in sections if s]

    # Split any oversized section further with recursive split
    chunks: list[dict[str, Any]] = []
    idx = 0
    for section in sections:
        if estimate_tokens(section) <= max_tokens:
            chunks.append(_make_chunk(doc, section, idx,
                                      {"chunking_rule": "confluence_heading"}))
            idx += 1
        else:
            sub = chunk_recursive(doc, {"chunk_size": max_tokens, "chunk_overlap": 100},
                                  rule_name="confluence_heading_overflow")
            # Reassign index and text from the oversized section
            for s in sub:
                chunks.append(_make_chunk(doc, s["text"], idx,
                                          {"chunking_rule": "confluence_heading_overflow"}))
                idx += 1
    return chunks or [_make_chunk(doc, text, 0, {"chunking_rule": "confluence_empty"})]


def chunk_google_drive(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Google Drive: recursive character split."""
    return chunk_recursive(doc, cfg, rule_name="google_drive_recursive")


def chunk_fireflies(doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fireflies: split at speaker turns (lines starting with 'Speaker:' pattern).

    We use a simple heuristic: a new chunk begins when a line looks like
    'Name:' at the start. If no turns are detected, treat as atomic.
    """
    max_tokens = cfg.get("max_tokens", 1024)
    text = doc["content"]
    lines = text.split("\n")

    turns: list[str] = []
    current: list[str] = []

    def looks_like_turn(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        # "Name: ..." or "Name (time): ..."
        colon_idx = stripped.find(":")
        if colon_idx < 1 or colon_idx > 40:
            return False
        prefix = stripped[:colon_idx]
        return prefix.replace(" ", "").replace("-", "").replace("_", "").isalpha()

    for line in lines:
        if looks_like_turn(line) and current:
            turns.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append("\n".join(current).strip())

    turns = [t for t in turns if t]

    if not turns:
        return [_make_chunk(doc, text, 0, {"chunking_rule": "fireflies_atomic"})]

    # Group turns into chunks up to max_tokens
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    idx = 0
    for turn in turns:
        trial = "\n".join(buffer + [turn])
        if estimate_tokens(trial) > max_tokens and buffer:
            chunks.append(_make_chunk(doc, "\n".join(buffer), idx,
                                      {"chunking_rule": "fireflies_turns"}))
            idx += 1
            buffer = [turn]
        else:
            buffer.append(turn)
    if buffer:
        chunks.append(_make_chunk(doc, "\n".join(buffer), idx,
                                  {"chunking_rule": "fireflies_turns"}))
    return chunks


def chunk_recursive(
    doc: dict[str, Any],
    cfg: dict[str, Any],
    rule_name: str = "recursive",
) -> list[dict[str, Any]]:
    """Character-based recursive splitter. Fallback and overflow handler."""
    chunk_size = cfg.get("chunk_size", 512) * 4  # chars (4 chars per token)
    overlap = cfg.get("chunk_overlap", 50) * 4
    text = doc["content"]

    if len(text) <= chunk_size:
        return [_make_chunk(doc, text, 0, {"chunking_rule": rule_name})]

    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end]
        chunks.append(_make_chunk(doc, piece, idx, {"chunking_rule": rule_name}))
        idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CHUNKERS = {
    "slack": chunk_slack,
    "gmail": chunk_gmail,
    "jira": chunk_jira,
    "linear": chunk_linear,
    "hubspot": chunk_hubspot,
    "github": chunk_github,
    "confluence": chunk_confluence,
    "google_drive": chunk_google_drive,
    "fireflies": chunk_fireflies,
}


def chunk_document(doc: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Dispatch to the correct chunker for this document's source type."""
    source = doc["source_type"]
    chunking_cfg = config["chunking"]

    if source in CHUNKERS:
        per_source = chunking_cfg["per_source"].get(source, {})
        return CHUNKERS[source](doc, per_source)

    # Unknown source: use default
    return chunk_recursive(doc, chunking_cfg["default"], rule_name="default_fallback")


def chunk_corpus(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the chunker to every document. Returns a flat list of chunks."""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, config))
    return all_chunks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.ingestion.loader import load_config, load_slice

    cfg = load_config()
    docs = load_slice(cfg)
    chunks = chunk_corpus(docs, cfg)

    print(f"Chunked {len(docs)} documents into {len(chunks)} chunks")
    print(f"Average chunks per doc: {len(chunks) / max(1, len(docs)):.2f}")

    # Chunks per source type
    from collections import Counter
    per_source = Counter(c["source_type"] for c in chunks)
    print("Chunks per source:")
    for source, n in sorted(per_source.items(), key=lambda x: -x[1]):
        print(f"  {source}: {n}")

    # Chunking rules used
    rules = Counter(c["metadata"]["chunking_rule"] for c in chunks)
    print("Chunking rules used:")
    for rule, n in sorted(rules.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {n}")