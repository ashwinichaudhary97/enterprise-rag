# Decision Journal

Every architectural and engineering decision, logged as it is made.

Format: date, decision, context, alternatives, rationale, reversibility.

---

## 2026-10-13 - Python 3.11 over 3.12

**Decision:** Use Python 3.11.9 (already installed) rather than upgrading to 3.12.

**Context:** Machine had 3.11.9 installed. Latest Python is 3.12/3.13. Needed to choose the target version for the venv.

**Alternatives considered:**
- Python 3.12 - newest stable
- Python 3.13 - bleeding edge
- Python 3.10 - older but stable

**Why 3.11:** ChromaDB, sentence-transformers, and torch 2.2.2 have prebuilt wheels for 3.11. Some lack 3.12/3.13 wheels, forcing source builds. 3.11 is the sweet spot for our stack.

**Reversibility:** Easy - recreate venv with different Python.

---

## 2026-10-13 - PowerShell over cmd

**Decision:** Use PowerShell as the development shell, not cmd.

**Context:** Initial terminal session was cmd. Needed to pick one for the rest of the project.

**Alternatives considered:**
- cmd - no execution policy, no setup cost
- Git Bash - Unix-style commands on Windows
- WSL - full Linux, but extra setup

**Why PowerShell:** Better Unicode handling (corpus has emojis, non-ASCII), better long-path support, aligns with modern Windows dev workflows, VS Code default. The one-time execution policy fix is worth it.

**Reversibility:** Easy - commands differ but project itself is shell-agnostic.

---

## 2026-10-13 - Pinned dependencies over floating

**Decision:** Pin every package version in requirements.txt (e.g. numpy==1.26.4 not numpy).

**Context:** Could install latest or pin to known-working versions.

**Alternatives considered:**
- Unpinned (numpy) - always latest
- Range pins (numpy>=1.26,<2) - flexible
- Exact pins - locked

**Why exact pins:** Reproducibility. A future pip install six months from now will produce the same environment. When a package breaks, we know exactly what version caused it.

**Reversibility:** Easy - edit requirements.txt.

---

## 2026-10-13 - CPU-only torch over CUDA build

**Decision:** Install torch==2.2.2 from PyTorch's CPU-only wheel index.

**Context:** Default pip install torch downloads the CUDA build (~2 GB) even when no NVIDIA GPU is present.

**Alternatives considered:**
- Default CUDA build - 2 GB waste, unused
- Older torch version - compatibility risk
- Install on Kaggle only - loses local iteration

**Why CPU-only:** Machine has no NVIDIA GPU. CPU build is 200 MB, works everywhere, and Kaggle scale-runs can use its own CUDA build separately.

**Reversibility:** Easy - pip install torch==2.2.2 (no index URL) gets the CUDA build.

---

## Template for New Entries

## YYYY-MM-DD - Short title

**Decision:** What was decided.

**Context:** Why this needed deciding.

**Alternatives considered:**
- Alt 1
- Alt 2

**Why this choice:** Rationale.

**Reversibility:** Easy / Medium / Hard.


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


---

## 2026-10-13 - Ingestion: Observed scaling across 100 / 1K / 10K docs

**Decision:** Freeze the working index at 10,000 documents after validating the pipeline at 100 and 1,000.

**Context:** The pipeline was run at three slice sizes to verify that (a) the code works end-to-end, (b) it scales sub-linearly, and (c) it produces a usable index. All timings are from the local i5-10210U / 8 GB / CPU-only machine.

**Observed timings:**

| Stage | 100 docs | 1,000 docs | 10,000 docs | Scaling |
|-------|----------|------------|-------------|---------|
| Load | 207.6s | 151.7s | 163.1s | O(1) — fixed |
| Chunk | 0.06s | 0.24s | 0.56s | ~linear |
| Embed | 37.5s | 149.4s | 1024.3s | sub-linear |
| Store | 3.6s | 10.6s | 101.3s | ~linear |
| **Total** | **248.9s** | **311.8s** | **1289.2s** | **sub-linear** |

**Derived rates:**

| Metric | 100 docs | 1,000 docs | 10,000 docs |
|--------|----------|------------|-------------|
| Chunks | 119 | 1,291 | 12,580 |
| Chunks per doc | 1.19 | 1.29 | 1.26 |
| s/chunk (embed) | 0.315 | 0.116 | 0.081 |

**Why embed time is sub-linear:** The first batch through MiniLM includes model warmup (graph construction, CPU dispatch cache). On 100 docs, that warmup dominates. On 10,000 docs, warmup is amortized across 394 batches. The true steady-state cost at 10K is ~0.08 s/chunk.

**Why load time is O(1):** The Hugging Face dataset is a single Parquet file that is memory-mapped regardless of slice size. Slice selection is applied after load. The ~150-200s is fixed cost, paid once per fresh Python process, eliminated on subsequent runs by the JSON slice cache (0.35s).

**Implication for the 50K Kaggle run:** At ~0.08 s/chunk on CPU, 50K would take ~1.5 hours locally. On Kaggle's T4 GPU, expect ~5-15 minutes due to ~20x GPU acceleration of the embedding step. The 50K run is feasible within Kaggle's 12-hour session limit.

**Implication for the 500K full corpus:** Not feasible locally. Would take ~20 hours of embedding on this CPU. Kaggle with a GPU could do it in ~1-2 hours, but is out of scope for this project.

**Reversibility:** Easy - change slice.size in configs/ingestion.yaml and re-run. Cache key changes, new cache written, index rebuilt.
