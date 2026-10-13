"""Append observed ingestion scaling numbers to docs/DECISIONS.md.

These are real measurements from the 100 / 1K / 10K runs on the local
machine. They inform the FINDINGS document and the 50K Kaggle estimate.
"""

from pathlib import Path

new_entry = """

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
"""

journal_path = Path("docs/DECISIONS.md")
existing = journal_path.read_text(encoding="utf-8")
journal_path.write_text(existing + new_entry, encoding="utf-8")
print(f"Appended {len(new_entry)} characters to {journal_path}")
print(f"New total size: {len(existing) + len(new_entry)} characters")