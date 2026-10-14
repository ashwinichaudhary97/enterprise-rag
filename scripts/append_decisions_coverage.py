"""Append the coverage finding to docs/DECISIONS.md.

Documents the discovery that a 10K stratified slice of the 512K-document
benchmark has only 3.4% coverage of ground-truth documents, making the
Phase 2 eval metrics meaningless. Records the decision to rebuild the
working index at full corpus size on Kaggle.
"""

from pathlib import Path

new_entry = """

---

## 2026-10-14 - Phase 2: Working corpus changed from 10K to 512K

**Decision:** Rebuild the working index at full corpus size (511,962 documents -> ~640,000 chunks), replacing the 10K slice. The 10K index is retained as a development index for fast iteration.

**Context:** Phase 2's first eval ran all four retrieval configs (dense, BM25, hybrid RRF, hybrid+rerank) against the 500 benchmark questions. The results were:

| Config | Recall@5 | nDCG@5 | MRR |
|--------|----------|--------|-----|
| hybrid_rerank | 0.0129 | 0.0142 | 0.0220 |
| hybrid | 0.0126 | 0.0109 | 0.0142 |
| bm25 | 0.0119 | 0.0136 | 0.0191 |
| dense | 0.0062 | 0.0062 | 0.0100 |

These numbers are 30-50x below any reasonable retrieval baseline. Investigation showed a structural cause.

**Diagnosis:** We ran a coverage analysis: for each of the 470 questions with ground truth, we checked whether at least one expected document is present in our 10K index. Result:

- Questions with ground truth: 470
- Questions with at least one expected doc in index: 16
- Coverage: 3.4%

Coverage by question type ranged from 0.0% (semantic, conflicting_info, miscellaneous) to 30.0% (completeness). The theoretical coverage for a random 10K sample of 512K docs on a single-doc question is ~2%, which matches the observed basic-question coverage of 1.7%.

The retrieval pipeline is not broken. The corpus is incomplete relative to the benchmark. No retrieval method can find a document that is not in the index. The metrics were measuring sampling luck, not retrieval quality.

**Alternatives considered:**

- Filter to only the 16 covered questions. Rejected: N=16 is too small for meaningful metrics.
- Rebuild at 50K on Kaggle (~10% coverage). Rejected: still 90% unreachable.
- Rebuild at 160K on Kaggle (~55% coverage). Rejected: half the benchmark still unreachable; the covered subset would be biased.
- Rebuild at 512K on Kaggle (full corpus, ~100% coverage). **Chosen.**
- Accept the 10K finding as the deliverable. Rejected: the project's whole downstream (access control, generation, routing) requires meaningful retrieval baselines to work against.

**Why 512K:** The benchmark was designed for the full corpus. Evaluating at less than 100% coverage produces a leaderboard whose numbers cannot be compared to any published system on this benchmark.

**Why Kaggle:** Local hardware (8 GB RAM, no GPU) cannot build or query a 20 GB index. Kaggle provides a T4/P100 GPU and ~29 GB RAM. Estimated full-corpus ingestion: ~1.5 hours on Kaggle GPU vs ~16 hours on local CPU.

**Why retain the 10K index:** Development index for fast iteration (30s eval vs ~1.5 hours). The 512K index is used for the final leaderboard and downstream phases.

**The 10K scaling data is not wasted.** All prior measurements (load O(1), embed sub-linear, store linear) remain valid and are recorded in earlier journal entries.

**Methodological finding (for FINDINGS.md):** Small stratified slices cannot evaluate large benchmarks when the questions have sparse ground truth. The 3.4% coverage metric is itself a publishable finding: it quantifies the gap between "the corpus looks representative" (source distribution matched) and "the corpus covers the questions" (it does not). Any project that samples a subset of a benchmark corpus should run this coverage check before reporting metrics.

**Reversibility:** Medium - rebuilding requires Kaggle session time and downloading a ~20 GB index. Reverting to 10K is a config change.
"""

journal_path = Path("docs/DECISIONS.md")
existing = journal_path.read_text(encoding="utf-8")
journal_path.write_text(existing + new_entry, encoding="utf-8")
print(f"Appended {len(new_entry)} characters to {journal_path}")
print(f"New total size: {len(existing) + len(new_entry)} characters")