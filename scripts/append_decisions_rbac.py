"""Append Phase 3 RBAC design decisions to docs/DECISIONS.md.

Records the design choices made in docs/RBAC_DESIGN.md. Each entry
follows the standard journal format: decision, context, alternatives,
rationale, reversibility.

Includes a duplicate guard: if the first or last RBAC entry is already
present, the script exits without writing.
"""

from pathlib import Path

GUARD_STRING_FIRST = "## 2026-10-15 - RBAC: Four-tier clearance hierarchy"
GUARD_STRING_LAST = "## 2026-10-15 - RBAC: Query-clustered paired bootstrap"

new_entry = """

---

## 2026-10-15 - RBAC: Four-tier clearance hierarchy

**Decision:** Use four access levels (`public < internal < confidential < restricted`) with an explicit total order.

**Context:** The RBAC layer needs a way to compare user clearance with document sensitivity. Options ranged from binary (public/private) to many-tier.

**Alternatives considered:**
- Binary public/private - too coarse; no room for manager vs. analyst distinction
- Three tiers (public/internal/confidential) - simpler, but no "restricted" top tier
- Four tiers with total order - deliberate experimental design
- Five or more tiers - over-engineered; diminishing returns

**Why four tiers:** This is a deliberate experimental design inspired by common enterprise classification schemes, not a claim that four tiers is a universal standard. Four levels give enough granularity for the four demo roles (intern/analyst/manager/admin) to be meaningfully different. The total order (`public < internal < confidential < restricted`) is explicit so the filtering predicate is unambiguous.

**Reversibility:** Medium - changing the tier count requires re-running Phase 3 experiments that depend on the affected assignments.

---

## 2026-10-15 - RBAC: SHA-256 for deterministic metadata assignment

**Decision:** Assign access_level and department via SHA-256 digest of a canonical input string, not Python's built-in `hash()`.

**Context:** Metadata must be reproducible across runs, machines, and Python processes. Python's `hash()` for strings is randomized between processes (PYTHONHASHSEED), producing different assignments on different runs.

**Alternatives considered:**
- Python `hash()` - not reproducible; process-randomized
- `random.seed(fixed)` + random.choice - reproducible if seed is fixed, but relies on Python's Mersenne Twister implementation stability across versions
- SHA-256 of canonical input - stable across all environments
- MD5 or SHA-1 - stable but cryptographically deprecated

**Why SHA-256:** Same doc_id always produces the same digest, on any machine, in any Python version. The algorithm is fully specified in RBAC_DESIGN.md §6.1: canonical input `rbac-v1:access:<doc_id>` and `rbac-v1:department:<doc_id>`, UTF-8 encoding, first 8 digest bytes as unsigned big-endian integer, divide by 2^64, cumulative weight mapping. No randomness, no seed.

**Reversibility:** Hard - changing the hash algorithm or byte selection invalidates results that depend on the affected assignments. Analyses that do not depend on those assignments are unaffected.

---

## 2026-10-15 - RBAC: Document-level permission propagation

**Decision:** Assign permissions at the document level, then propagate to every chunk belonging to that document.

**Context:** A document splits into multiple chunks during ingestion. If permissions were assigned per chunk, chunks of the same document could end up with different access levels.

**Alternatives considered:**
- Chunk-level assignment - creates inconsistent permissions within a document
- Document-level assignment with propagation - permissions are consistent by construction
- Row-level permissions at query time - viable depending on storage and query architecture; trade-off between flexibility and per-query overhead

**Why document-level:** A document is one logical unit. A user either can access a document or cannot. Propagating the parent's permissions to all chunks preserves that invariant. It also assigns once per document rather than per chunk. The invariant is verified by tests before any evaluation runs.

**Reversibility:** Easy - the propagation step is one function; the assignment algorithm is unchanged.

---

## 2026-10-15 - RBAC: Pre-filtering and post-filtering as a designed comparison

**Decision:** Implement both pre-filtering (filter at DB before retrieval) and post-filtering (retrieve full corpus, then remove unauthorized). Compare them on the same queries and users.

**Context:** Pre-filtering is the secure default. Post-filtering preserves the full candidate pool but may return fewer than K authorized results.

**Alternatives considered:**
- Pre-filtering only - secure but may lose candidates under approximate retrieval
- Post-filtering only - risky (unauthorized docs enter retrieval pipeline); may return fewer than K results
- Prompt-based filtering - security failure; forbidden
- Pre + post comparison - measures the trade-off directly

**Why both:** Under the documented assumptions in RBAC_DESIGN.md §1 (same authorized corpus, same relevance labels, same scoring function, exact top-K, at least one authorized relevant document), pre-filtering should not reduce Authorized Recall@5 relative to the unfiltered authorized-scoring baseline. If pre-filtering does reduce it, the cause is investigated (approximate-index behavior, `where`-filter implementation, configuration) before attribution. Post-filtering is compared on the same query-user pairs to quantify returned-result loss and latency. Neither strategy is presented as universally correct.

**Reversibility:** Easy - the two strategies are separate code paths; we can pick one for production without losing the comparison data.

---

## 2026-10-15 - RBAC: Authorized Recall@5 as a new metric

**Decision:** Introduce Authorized Recall@5 alongside standard Recall@5.

**Context:** Standard Recall@5 counts retrieved documents that are relevant to the query. In an access-controlled system, some relevant documents are not accessible to the user. Counting them as retrieval successes is wrong for enterprise RAG.

**Alternatives considered:**
- Standard Recall@5 only - wrong denominator for enterprise
- Authorized Recall@5 only - loses the diagnostic view of the full ground truth
- Both, reported separately - measures security-aware retrieval AND diagnostic retrieval

**Why both:** Standard Recall@5 answers "did retrieval find the right documents anywhere." Authorized Recall@5 answers "did retrieval find the right documents the user is allowed to see." Both are informative. The metrics have different denominators, so their numerical difference is not treated as a security-cost measure; each is reported with its own denominator and undefined-pair count.

**Denominator rule:** Authorized Recall@5 is defined only for query-user pairs with at least one authorized relevant document. Pairs with zero authorized relevant documents are excluded from the denominator (undefined, not zero) and reported separately.

**Reversibility:** Medium - the metric is a core deliverable; removing it reduces the finding's novelty.

---

## 2026-10-15 - RBAC: v1 and v2 sensitivity distributions

**Decision:** Run the full evaluation against two metadata distributions: v1 (permissive) and v2 (restrictive).

**Context:** The permissions are synthetic. A finding that holds only under one distribution is distribution-dependent, not general.

**Alternatives considered:**
- v1 only - single distribution; cannot distinguish structural effects from distribution-specific ones
- v1 and v2 - tests whether conclusions are stable
- Three or more distributions - diminishing returns; time cost

**Why two:** Two distributions give one sensitivity data point without tripling the runtime. v2 is specified in full in RBAC_DESIGN.md §6.4 and frozen at commit. The v2 weights are not chosen after seeing v1 results.

**Reversibility:** Medium - adding a v3 requires re-running evaluations that depend on the affected assignments.

---

## 2026-10-15 - RBAC: Query-clustered paired bootstrap

**Decision:** Use a paired bootstrap clustered by query ID for confidence intervals, not a per-query-user-pair bootstrap.

**Context:** Multiple users evaluate the same 500 queries. Treating query-user pairs as independent observations overstates the effective sample size and produces artificially narrow confidence intervals.

**Alternatives considered:**
- Per-query-user-pair bootstrap - treats pairs as independent; wrong when users share queries
- Query-clustered bootstrap - resamples queries as units, retaining their users
- Paired t-test - assumes normality, not appropriate for paired differences in retrieval metrics

**Why clustered:** The unit of independence is the query, not the query-user pair. When a query is sampled, its results for all users are sampled together. This preserves the shared structure and produces honest uncertainty estimates.

**Parameters (frozen before evaluation):**
- Iterations: 1000
- Seed: 42
- Resampling unit: query ID
- Paired difference computed within each resample

**Reversibility:** Easy - the bootstrap is one analysis function.
"""

journal_path = Path("docs/DECISIONS.md")
existing = journal_path.read_text(encoding="utf-8")

if GUARD_STRING_FIRST in existing or GUARD_STRING_LAST in existing:
    raise RuntimeError("RBAC decisions already appear in DECISIONS.md")

journal_path.write_text(existing + new_entry, encoding="utf-8")
print(f"Appended {len(new_entry)} characters to {journal_path}")
print(f"New total size: {len(existing) + len(new_entry)} characters")