"""One-off script to append ingestion design decisions to docs/DECISIONS.md.

Run with: python scripts/append_decisions_ingestion.py
"""

from pathlib import Path

new_entries = """

---

## 2026-10-13 - Ingestion: Stratified sampling over random or sequential

**Decision:** Sample the working slice by stratifying across source types, not by taking the first N documents or sampling randomly.

**Context:** The full corpus has 9 source types with very different volumes (Slack 55.8%, Confluence 1.0%). The working slice must be representative so chunking rules for every source are exercised.

**Alternatives considered:**
- First N documents - simplest, but biased by dataset ordering
- Random N - unbiased in expectation, but can miss small source types by chance
- Stratified N - proportional sampling from each source
- Hand-picked N - controllable, but introduces researcher bias

**Why stratified:** Slack is 56% of the corpus. Random sampling at 1K might produce zero HubSpot documents, and then HubSpot chunking bugs would never surface. Stratification guarantees every source type appears in proportion, exercising all code paths.

**Reversibility:** Easy - change the sampling function.

---

## 2026-10-13 - Ingestion: Source-aware chunking over fixed-size

**Decision:** Split documents using source-specific rules (Slack threads stay together, Confluence splits at headings, etc.), not a uniform fixed-size window.

**Context:** Different enterprise document types have different semantic units. Slack threads are conversations; Jira issues are atomic; Confluence pages are hierarchical. Fixed-size chunking destroys these boundaries.

**Alternatives considered:**
- Fixed-size 512 tokens - simple, uniform, but splits threads and headings
- Semantic chunking (embedding-based splits) - highest quality, O(n^2) cost
- Paragraph-only - simple, ignores source structure
- Source-aware - respects each type's natural boundary

**Why source-aware:** A 512-token window will split a Slack thread mid-conversation, separating a problem from its resolution. Retrieval then returns incomplete evidence. Source-aware chunking preserves the natural units the benchmark was designed around.

**Reversibility:** Easy - chunking function is isolated.

---

## 2026-10-13 - Ingestion: MiniLM 384-dim over mpnet 768-dim

**Decision:** Use all-MiniLM-L6-v2 (384 dimensions) as the embedding model, not all-mpnet-base-v2 (768 dimensions) or an API model.

**Context:** Machine is CPU-only with 8 GB RAM. Embedding time dominates the ingestion pipeline. mpnet is ~3.5x slower on CPU than MiniLM.

**Alternatives considered:**
- all-MiniLM-L6-v2 (384d) - fast, small, standard starter model
- all-mpnet-base-v2 (768d) - 3-4 point recall gain, 3.5x slower, 5x larger model
- bge-small-en-v1.5 - modern, similar speed
- OpenAI text-embedding-3-small - best quality, API cost, network dependency

**Why MiniLM:** On an 8 GB CPU-only machine, the 3.5x speed penalty of mpnet is real: 10K docs would take 2-3 hours instead of 20-40 minutes. The recall gain (3-4 points absolute on MTEB) does not justify it at this stage. The Decision Journal records the upgrade path: if the 1K debug run shows recall below 0.70, we swap to mpnet before the 10K freeze.

**Reversibility:** Medium - requires re-embedding the entire corpus, but chunking and storage layers do not change.

---

## 2026-10-13 - Ingestion: ChromaDB over FAISS, Qdrant, or Weaviate

**Decision:** Use ChromaDB as the vector store.

**Context:** RBAC requires metadata filtering at the retrieval layer. The store must support attaching fields (access_level, department) to each chunk and filtering on them during query.

**Alternatives considered:**
- FAISS - fastest similarity search, but no metadata filtering
- ChromaDB - metadata filtering, simple API, single-file persistence
- Qdrant - native filtering, hybrid search, requires separate service
- Weaviate - enterprise-grade, heavy infrastructure
- Pinecone - managed cloud, cost, network dependency

**Why ChromaDB:** Metadata filtering is mandatory for the access control layer. FAISS cannot filter. Qdrant is stronger at scale but requires running a separate service. ChromaDB gives us filtering, persistence, and a simple API at the scale we operate (10K docs). If we scaled to 10M, we would revisit.

**Reversibility:** Medium - swapping stores requires rewriting the storage layer, but the data model is portable.

---

## 2026-10-13 - Ingestion: Batch size 32 over 8 or 128

**Decision:** Embed documents in batches of 32.

**Context:** 8 GB RAM ceiling. Batch size controls the trade-off between throughput and memory footprint during embedding.

**Alternatives considered:**
- Batch 8 - very safe, slow throughput
- Batch 32 - balanced
- Batch 128 - faster, higher memory risk
- Batch 512 - fastest, will OOM on 8 GB machine

**Why 32:** MiniLM at 384-dim uses roughly 5 MB per document in a forward pass. Batch 32 = ~160 MB peak memory, safe alongside the corpus and ChromaDB. Batch 128 would push 640 MB and risks paging on an 8 GB machine. Empirical choice, not dogma.

**Reversibility:** Easy - batch size is a config parameter.

---

## 2026-10-13 - Ingestion: On-disk persistence over in-memory

**Decision:** Persist the ChromaDB index to disk at data/indexes/chroma/.

**Context:** The index must survive shell restarts. Phases 2-6 all query against the same index and must produce comparable metrics.

**Alternatives considered:**
- In-memory - fast startup, lost on shell close
- On-disk - persists, reloadable in seconds
- Cloud (S3, GCS) - durable, network dependency, unnecessary locally

**Why on-disk:** Phases 2-6 run across many sessions. Rebuilding the index every session would cost 20-40 minutes each time. Persistence is required for reproducibility and iteration speed.

**Reversibility:** Easy - change the persist_directory argument.

---

## 2026-10-13 - Ingestion: Staged workflow (100 -> 1K -> 10K)

**Decision:** Build the pipeline at 100 documents, debug at 1K (stratified), run once at 10K (stratified), freeze.

**Context:** Pipeline code is identical across all three stages. Only the input slice changes. Compute cost scales roughly linearly with slice size, but iteration speed is what determines how fast bugs get found.

**Alternatives considered:**
- Straight to 10K - wastes time on slow iterations
- Straight to 1K - misses the fast build loop
- 100 -> 1K -> 10K - progressive
- 100 -> 10K - skips edge-case surfacing at 1K

**Why staged:** At 100 docs, iteration takes seconds. Structural bugs surface immediately. At 1K, rare source types appear and edge cases surface in 2-4 minutes. At 10K, we run once and freeze. Total debugging time is roughly 1 hour instead of 6-8 hours if we started at 10K.

**Reversibility:** Easy - slice size is a parameter.
"""

journal_path = Path("docs/DECISIONS.md")
existing = journal_path.read_text(encoding="utf-8")
journal_path.write_text(existing + new_entries, encoding="utf-8")
print(f"Appended {len(new_entries)} characters to {journal_path}")
print(f"New total size: {len(existing) + len(new_entries)} characters")