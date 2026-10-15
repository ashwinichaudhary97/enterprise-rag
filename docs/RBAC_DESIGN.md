# RBAC Design for Access-Controlled Retrieval Evaluation

**Version:** v1.0
**Status:** Frozen after review
**Date:** 2026-10-15
**Author:** Ashwini Chaudhary
**Project:** Enterprise RAG Platform

This document defines the schema, policy, evaluation protocol, and reproducibility contract for Phase 3 (Access Control). It is written before implementation. Schema, policy, and assignment changes require a version bump and re-running affected experiments.

---

## 1. Research Hypotheses

The following hypotheses are stated before the experiments. They are falsifiable; either outcome is valid and must be reported honestly.

**H1 — Access-control filtering affects retrieval quality differently across retrieval strategies.**

Dense, BM25, and hybrid retrieval may be affected by access filtering by different amounts. The relative ranking of strategies may change when filtering is applied.

**H1 metric definition:**

The primary comparison is Authorized Recall@5 under filtered vs. unfiltered retrieval. Both conditions use the **same denominator**: the number of ground-truth relevant documents the user is authorized to see.

- **Fixed denominator:** Filtering cannot change the Authorized Recall@5 denominator. It remains the number of ground-truth relevant documents the user is authorized to access.
- **Expected pre-filtering behavior:** Under exact retrieval with fixed relevance scores and identical ranking logic, pre-filtering removes unauthorized candidates before ranking, allowing authorized candidates to move up. Under the assumptions of (a) the same authorized corpus, (b) the same relevance labels, (c) the same scoring function, (d) exact top-K ranking without approximation, with identical scoring and ranking semantics over the authorized candidates, and (e) at least one authorized relevant document per query-user pair, pre-filtering should not reduce Authorized Recall@5 relative to the unfiltered authorized-scoring baseline. The unfiltered baseline retrieves globally, then counts only authorized relevant documents among the global top five; pre-filtering ranks only authorized documents. Both strategies use the same fixed denominator of ground-truth relevant documents authorized for that query-user pair.
- **Three distinct effects are reported separately:**
  - **Post-filtering result loss:** Unauthorized candidates occupy top-5 slots and are removed after retrieval, potentially leaving fewer than five authorized results. Measure this using the number of authorized results returned and the fraction of requests returning fewer than five results. Report Authorized Recall@5 separately.
  - **Pre-filtering versus post-filtering:** Compare Authorized Recall@5, authorized result counts, and latency using the same query-user pairs, corpus, ground truth, retrieval configuration, and requested K. Do not silently refill post-filtered results.
  - **Unexpected pre-filtering degradation:** If pre-filtering produces lower Authorized Recall@5 than the unfiltered authorized-scoring baseline, investigate before attributing the difference to access control. Potential causes include approximate-index behavior, incorrect or inconsistent `where`-filter implementation, candidate-generation differences, ranking changes, and configuration mismatches. Report the observed result and investigation transparently.

**H2 — The effect varies with query characteristics and the fraction of relevant documents the user can access.**

For each query-user pair, compute `authorized_fraction = (# authorized relevant docs) / (# all relevant docs)`. Group pairs by **mutually exclusive** buckets:

| Bucket | Range |
|---|---|
| B1 | [0.00, 0.25) |
| B2 | [0.25, 0.50) |
| B3 | [0.50, 0.75) |
| B4 | [0.75, 1.00] |

The top bucket is closed at 1.00. All other buckets are half-open. No value belongs to two buckets.

Report mean Authorized Recall@5 per bucket. **Monotonicity is a hypothesis to test, not a result assumed by the design.** If the observed relationship is flat, non-monotonic, or in the opposite direction, that is the finding.

**H3 — Adaptive routing improves the quality–latency trade-off over the best fixed retrieval strategy under the same access policy.**

The router is evaluated against fixed retrieval strategies using the same queries, users, authorization policy, and ground truth. Quality and latency are reported separately. The router is considered beneficial if it:

- improves retrieval quality at comparable latency, or
- reduces latency at comparable quality, or
- demonstrates a favorable trade-off across multiple operating points.

**Fairness safeguard:** The "best fixed strategy" is selected on a validation split. Any router tuning also uses validation data. The final comparison uses a held-out test split. Selecting the best fixed strategy or tuning the router on the test queries would bias the comparison.

H3 is tested in Phase 6, not Phase 3.

---

## 2. Field Schema

Every document in the 512K corpus receives two synthetic permission fields. Permissions are assigned at the document level and propagated to every chunk belonging to that document.

| Field | Type | Values | Purpose |
|---|---|---|---|
| `document_id` | string | Existing `doc_id` | Stable identifier for the parent document |
| `chunk_id` | string | Existing chunk ID | Unique identifier for each chunk |
| `access_level` | enum | `public`, `internal`, `confidential`, `restricted` | Synthetic document clearance level |
| `department` | enum | `engineering`, `sales`, `marketing`, `hr`, `finance`, `support` | Synthetic document department |
| `metadata_policy_version` | string | `"v1"` | Policy version |
| `metadata_assignment` | string | `"synthetic"` | Provenance: labels are simulated, not supplied by the benchmark |

`document_id` and `chunk_id` already exist in the system. Reuse the existing identifiers; do not generate replacements unnecessarily.

`metadata_policy_version` records which policy version was used. `metadata_assignment` records label provenance. Neither alone guarantees reproducibility; the stable assignment algorithm, assignment version, distribution parameters, source-type adjustment rules, and any seed must be recorded in the experiment manifest.

Chunks inherit the parent document's permission values. A chunk must never have a different `access_level` or `department` from its parent document. This is a hard invariant and is tested before evaluation.

---

## 3. Clearance Hierarchy

`access_level` values form a total order:

    public < internal < confidential < restricted

A user with clearance `X` may access documents at level `X` and below.

**Clearance alone does not grant access to every non-public document.** Department membership is a required second constraint for all non-public documents. A user with `confidential` clearance cannot access a `confidential` document in a department they do not belong to.

Public documents are exempt from the department constraint. See Section 5 for the complete policy.

---

## 4. User Definitions

Four demo users are defined. Clearance and department membership are independent attributes.

| User | Clearance | Departments | Notes |
|---|---|---|---|
| `admin` | `restricted` | All | Can access every document |
| `manager` | `confidential` | `engineering`, `sales` | Any level up to `confidential` in either department, plus all public documents |
| `analyst` | `internal` | `engineering` | Any level up to `internal` in `engineering`, plus all public documents |
| `intern` | `public` | All | Public documents only, across all departments |

**Explicit rule:** public documents are accessible across all departments. A public document assigned to `sales` is visible to an `analyst` whose department membership is `engineering`.

For policy implementation, represent "all departments" explicitly (e.g., as a documented wildcard or a complete set). **Do not use an empty list to mean both "all departments" and "no departments."**

---

## 5. Filtering Policy

The policy is applied at query time. The authorization predicate is:

For a user with clearance `C` and department membership `D`:

1. If the document's `access_level` is `public`, it is accessible regardless of department.
2. Otherwise, the document is accessible only if `access_level <= C` AND `department ∈ D`.

This predicate is the source of truth. Database filters must implement the same behavior, and independent policy tests must verify the result.

**Two strategies are implemented and compared:**

| Strategy | Description | Security property |
|---|---|---|
| Pre-filtering | Restrict the searchable corpus to authorized documents before similarity search (e.g., ChromaDB `where` filter). | Unauthorized documents must not enter the candidate set. |
| Post-filtering | Retrieve top-K candidates from the full corpus, then remove unauthorized candidates before returning results. | Unauthorized candidates may enter the internal retrieval pipeline, but must never be returned to the user or exposed through logs, traces, citations, or debugging output accessible to that user. |

**Primary implementation:** pre-filtering.

**Comparison protocol:** pre-filtering and post-filtering use the same query-user pairs, corpus, ground truth, retrieval configuration, and requested K.

**If post-filtering returns fewer than K authorized results**, record the returned count; **do not silently increase K or refill results.** Any additional candidate-fetching variant must be separately named and evaluated.

**Hard security rule:** neither strategy may expose an unauthorized document to the user. Any exposure is a security failure, must be logged safely, and halts the affected experiment pending investigation. Internal evaluation code may inspect candidate IDs only within a trusted evaluation boundary; user-facing output must contain authorized results only.

---

## 6. Metadata Assignment Method

Permission assignment must be deterministic, reproducible, and versioned.

### 6.1 Stable assignment algorithm

The assignment algorithm is fully specified below. There are no open implementation choices.

**Canonical inputs:**

- Access level: `rbac-v1:access:<doc_id>`
- Department: `rbac-v1:department:<doc_id>`

For v2, the prefix is `rbac-v2:` instead of `rbac-v1:`. All other parameters are unchanged.

**Encoding:** UTF-8.

**Digest:** SHA-256.

**Integer conversion:** the first 8 bytes of the digest, interpreted as unsigned big-endian, producing an integer in `[0, 2^64)`.

**Mapping:** divide the integer by `2^64` to obtain a value `u` in `[0, 1)`. Assign the first category whose cumulative normalized weight exceeds `u`. Categories are ordered as listed in the distribution tables in §6.3 and §6.4.

**Source-type adjustment:** before mapping, apply the source-type multipliers to the base weights and renormalize each distribution independently (see §6.2). Apply the adjustment before computing cumulative weights.

**Missing or unknown `source_type`:** use neutral multipliers (1.00) for all categories in both distributions.

**Seed:** the assignment is fully hash-derived. No random generator is used. The manifest records `seed: not applicable (stable hash assignment)`.

The canonical input format, encoding, digest, byte selection, byte order, integer conversion, interval mapping, source-type adjustments, and distribution weights together define the reproducibility contract. Any change to any of these requires a version bump.

### 6.2 Source-type adjustment (v1)

Source type is a weak prior, not a hard rule. Slack documents are somewhat more likely to be `internal`. Gmail is somewhat more likely to be `confidential`. HubSpot documents are somewhat more likely to be in the `sales` department (department-weight adjustment, table below). **No source type maps deterministically to a permission label.**

**Access-level adjustment (v1):**

| source_type | Multiplier on `internal` weight | Multiplier on `confidential` weight |
|---|---|---|
| slack | 1.10 | 0.90 |
| gmail | 0.95 | 1.15 |
| hubspot | 0.90 | 1.10 |
| github | 1.05 | 0.95 |
| confluence | 1.00 | 1.00 |
| jira | 1.05 | 0.95 |
| linear | 1.05 | 0.95 |
| fireflies | 1.00 | 1.00 |
| google_drive | 0.90 | 1.10 |

**Department adjustment (v1):**

| source_type | Multiplier on `sales` weight | Multiplier on `engineering` weight |
|---|---|---|
| hubspot | 1.40 | 0.70 |
| github | 0.80 | 1.20 |
| jira | 0.90 | 1.15 |
| linear | 0.90 | 1.15 |
| gmail | 0.95 | 1.05 |
| slack | 0.95 | 1.05 |
| confluence | 1.00 | 1.00 |
| fireflies | 1.00 | 1.00 |
| google_drive | 1.00 | 1.00 |

Multipliers are applied to the access-level base weights (§6.3) and renormalized to sum to 1 before sampling. **Access-level adjustment and department adjustment are applied independently — each adjusts its own base distribution and normalizes separately. They do not multiply each other.**

**Access-level table — omitted categories:** only `internal` and `confidential` multipliers are listed per source type. For `public` and `restricted`, the multiplier is `1.00` for every source type. For any source type not listed in the table, all four access-level multipliers are `1.00`.

Multipliers are applied to the department base weights (§6.3) and renormalized to sum to 1 before sampling. Department adjustment is applied independently of access-level adjustment.

**Department table — omitted categories:** only `sales` and `engineering` multipliers are listed per source type. For `marketing`, `hr`, `finance`, and `support`, the multiplier is `1.00` for every source type. For any source type not listed in the table, all six department multipliers are `1.00`.

### 6.3 Distribution v1

**Access level weights (base):**

| access_level | Weight |
|---|---|
| public | 15% |
| internal | 55% |
| confidential | 22% |
| restricted | 8% |

**Department weights (base):**

| department | Weight |
|---|---|
| engineering | 40% |
| sales | 20% |
| marketing | 12% |
| hr | 10% |
| finance | 10% |
| support | 8% |

These are target weights. The realized corpus distribution may differ slightly because assignments are made deterministically. **Report the realized document-level distribution** and verify that every chunk inherits the correct document assignment.

### 6.4 Sensitivity distribution v2

v2 uses a more restrictive distribution than v1. Its full specification is committed here before running v2.

**Access level weights (v2):**

| access_level | Weight |
|---|---|
| public | 8% |
| internal | 42% |
| confidential | 32% |
| restricted | 18% |

**Department weights (v2):**

| department | Weight |
|---|---|
| engineering | 35% |
| sales | 20% |
| marketing | 12% |
| hr | 12% |
| finance | 13% |
| support | 8% |

**Source-type adjustments (v2):** Use the access-level and department adjustment tables specified in §6.2, applied to the corresponding v2 base distributions in §6.4. Access-level and department adjustments are applied independently, and each distribution is renormalized separately. Unspecified source types use neutral multipliers (1.00) for all access levels and departments.

**Rationale:** more restrictive distribution. Higher fraction of `confidential` and `restricted` documents than v1. Tests whether conclusions from v1 depend on a permissive distribution.

**Rule:** v2 weights are frozen at commit. They must not be adjusted after seeing results.

Run v1 and v2 and report which conclusions are stable across both and which are distribution-dependent. If only one version is completed, state that limitation explicitly.

---

## 7. Evaluation Protocol

### 7.1 Unfiltered baseline

Measure dense, BM25, hybrid, and hybrid-plus-reranking on the same benchmark questions without access filtering. Existing Phase 2 results are the reference baseline, but they must be checked for consistent corpus version, query split, ground truth, K, metric implementation, and retrieval configuration before reuse.

The benchmark is described as having 500 questions. **Record the actual question count and any exclusions in every run.**

### 7.2 Access-controlled evaluation

Run the same retrievers for each of the four users using the policy in Section 5. Pre-filtering and post-filtering must use the same query-user pairs, corpus, ground truth, retriever configuration, and K.

### 7.3 Query-user authorization labels

Compute these Boolean fields for each query-user pair:

- `has_authorized_relevant`: at least one ground-truth relevant document is authorized for this user.
- `has_unauthorized_relevant`: at least one ground-truth relevant document is not authorized for this user.

Derive one mutually exclusive label:

| `has_authorized_relevant` | `has_unauthorized_relevant` | Label | Interpretation |
|---|---|---|---|
| true | false | `authorized_answerable` | Relevant ground truth exists and all relevant documents are authorized |
| false | true | `unauthorized_only` | Relevant ground truth exists but none is authorized; correct behavior is no relevant result or a safe refusal |
| true | true | `mixed` | Both authorized and unauthorized relevant documents exist |
| false | false | `no_relevant_ground_truth` | No ground-truth relevant document is available to classify |

`authorized_answerable` and `mixed` both have at least one authorized relevant document. Do not double-count them in coverage calculations.

### 7.4 Metric definitions

**Evaluation unit and deduplication:** The ground-truth relevance unit is the parent document identified by `doc_id`. Retrieved chunks must be mapped to their parent document IDs, with duplicate parent IDs deduplicated while preserving the highest-ranked occurrence. Document-level Recall@5 is computed over the top five unique parent documents. Permission checks are performed using the parent document's assigned permissions, and every chunk must inherit those permissions.

Report chunk-level retrieval counts separately if they are needed for engineering diagnostics. The implementation must verify that `expected_doc_ids` and retrieved parent document IDs use the same identifier format.

| Metric | Definition | Denominator / reporting rule |
|---|---|---|
| Authorized Recall@5 | Authorized relevant documents retrieved in top-5, divided by the number of authorized relevant ground-truth documents | Compute only for query-user pairs with at least one authorized relevant document; report excluded/undefined pair count |
| Global Recall@5 | Relevant documents retrieved in top-5, divided by all relevant ground-truth documents, regardless of authorization | Diagnostic only; never used to justify returning unauthorized results |
| Answerable-query coverage | Fraction of all evaluated query-user pairs with at least one authorized relevant document | Count `authorized_answerable` and `mixed` once each; denominator is all evaluated query-user pairs |
| Unauthorized exposure count | Number of unauthorized documents returned to a user | Absolute count; target is zero |
| Returned authorized result count | Number of authorized results returned for each query-user pair | Report distribution and fraction of requests returning fewer than K results |
| Latency (p50, p95) | Time to apply authorization and retrieve/filter results | Milliseconds; measure consistently and state what stages are included |

A query-user pair with zero authorized relevant documents has **undefined Authorized Recall@5**. Exclude it from the Authorized Recall@5 denominator and report the number of such pairs; do not silently assign zero.

Authorized Recall@5 and returned-result counts are computed as separate metrics. They are never combined or interchanged. Both are reported in every result table with their denominators labeled.

### 7.5 H1 and H2 analysis

Report Authorized Recall@5 and Global Recall@5 separately. They answer different questions and have different denominators.

**For H1:** report the following comparisons separately, all on the same query-user pairs and with Authorized Recall@5 using the same fixed denominator in every comparison:

1. **Unfiltered global top-5 vs. pre-filtered top-5.** Measures the effect of restricting the searchable corpus before retrieval.
2. **Unfiltered global top-5 followed by post-filtering vs. pre-filtered top-5.** Measures the effect of applying permissions before retrieval vs. after.
3. **Pre-filtering vs. post-filtering for returned-result counts and latency.** Measures how often post-filtering returns fewer than K results, and the latency cost of each strategy.

Do not subtract scores with incompatible denominators. Do not report an aggregate "filtering degradation" number that mixes these three comparisons. Each comparison answers a different question and is reported on its own.

Report undefined Authorized Recall@5 pairs separately in every comparison.

**For H2:** calculate the fraction of ground-truth relevant documents authorized for each query-user pair where ground truth is non-empty. Analyze outcomes by that fraction using the mutually exclusive buckets defined in Section 1. When reliable query category labels exist, analyze by category as well. If query categories are not provided or validated, do not invent them after looking at the results; report the limitation.

For each authorized-fraction bucket, report the number of query-user pairs, the number with defined Authorized Recall@5, the number with undefined Authorized Recall@5, and the mean Authorized Recall@5 over defined pairs only. Report answerable-query coverage separately. Do not assign zero Authorized Recall@5 to pairs with no authorized relevant documents.

**Correct denial is not a retrieval failure** when no relevant ground-truth document is authorized. A retrieval failure occurs when authorized relevant documents exist but the system fails to retrieve them. Security correctness and retrieval quality must be reported separately.

---

## 8. Filtering Correctness and Security Tests

Independent of ChromaDB, test the authorization predicate against a small, explicit fixture of documents and users covering:

- Public documents from other departments
- Confidential documents outside a user's department membership
- Restricted documents (accessible only to admin under the defined demo roles)
- Users with public-only clearance
- Users with multiple departments and clearance constraints
- Documents exactly at the user's clearance boundary
- Empty department membership (must mean no non-public department access, not unrestricted access)
- Explicit all-department membership representation

Test the policy function independently, then verify that the database filter returns the same authorized set for each fixture case. Verify document-to-chunk permission inheritance across the corpus, **including missing or conflicting parent metadata**.

Every returned document is checked against the policy. Zero unauthorized documents is the passing condition. A failure halts the experiment.

Run these tests on every evaluation run, not just once.

---

## 9. Statistical Analysis

**Paired comparisons:** compare pre-filtering and post-filtering on the same query-user pairs. Store per-query, per-user results so the paired analysis can be reconstructed.

**Confidence intervals:** report 95% confidence intervals for paired differences in key metrics, including Recall@5 where defined.

**Clustered bootstrap:** because multiple users may share a query, resample query IDs as clusters. When a query ID is sampled, retain its associated users and their results together. Compute the paired difference within each resample. This avoids treating query-user pairs that share the same query as independent observations.

Record the number of bootstrap resamples (1000) and the random seed (42) used for statistical resampling. Both values are frozen for Phase 3.

**Reporting:** every headline result includes a point estimate, uncertainty interval, and sample size. Report undefined metric counts and any exclusions. Avoid claiming a meaningful difference solely because point estimates differ.

---

## 10. Reproducibility Manifest and Limitations

### 10.1 Manifest (recorded per run)

| Field | Value to record |
|---|---|
| Corpus version | Corpus identifier and version; current target: 512K corpus / 642,486 chunks, subject to verification |
| Index ID | `ashwini786/enterprise-rag-chroma-512k` |
| Questions | EnterpriseRAG-Bench; target 500 questions, verify actual count |
| Ground truth | `expected_doc_ids` field; record schema/version and any normalization |
| Metadata policy version | v1 and, for sensitivity analysis, v2 |
| Assignment algorithm | Canonical inputs (`rbac-v1:access:<doc_id>`, `rbac-v1:department:<doc_id>`), SHA-256, first 8 bytes unsigned big-endian, divide by `2^64`, cumulative weight mapping, source-type multipliers, weights |
| Assignment seed | Not applicable (stable hash assignment) |
| Retriever configs | Dense (MiniLM, 384 dimensions), BM25 (bm25s split/config), hybrid (RRF configuration), reranker (ms-marco-MiniLM-L-6 configuration) |
| Query split | Exact query IDs for validation and held-out test sets |
| Random seeds | Statistical bootstrap: 1000 iterations, seed 42. Other seeds: none (assignment uses stable hash). |
| Metric definitions | Section 7, including denominator rules and undefined cases |
| Software versions | Python, OS/runtime, key package versions, and `requirements.txt` or lockfile |
| Run identifier | UUID per evaluation run |
| Results / artifacts | Config snapshot, per-query-user outputs, aggregate metrics, test results, logs, and timestamps |

**Do not treat an index name or a target corpus count as proof that the loaded corpus matches it.** Record and verify the actual index/corpus identifiers, document count, chunk count, and benchmark question count at run time.

### 10.2 Limitations to report in FINDINGS

1. **Synthetic permissions.** The benchmark does not supply these access labels. Findings apply to the simulated policy and assignment, not automatically to real enterprise security.
2. **Assignment distributions.** Report the exact v1 and v2 distributions tested. State which conclusions remain stable and which depend on the distribution. If only one is tested, acknowledge this.
3. **Single embedding model.** Dense retrieval uses MiniLM-L6. Stronger models such as BGE-M3 or MPNet have not been tested unless additional experiments are completed.
4. **No real IAM integration.** Demo users are defined locally; real deployments would integrate with an identity provider and enforce permissions throughout the serving stack.
5. **Benchmark scope.** Findings apply to the tested EnterpriseRAG-Bench corpus and queries. Generalization to other enterprise corpora is not established.
6. **Synthetic policy assumptions.** The chosen clearance hierarchy, departments, and role definitions are experimental assumptions, not evidence of any particular organization's actual access policy.

### 10.3 Freeze rule

Changes to the schema (Section 2), clearance hierarchy (Section 3), users (Section 4), authorization policy (Section 5), or assignment algorithm/distribution (Section 6) require a version bump and re-running all affected experiments.

Evaluation or analysis additions in Sections 7–9 may be added if they are documented and prior results remain valid. **Any change that alters metric definitions, denominators, query selection, statistical procedures, or result interpretation must be recorded, and affected analyses must be rerun.** Additive changes are not automatically harmless.

---

## Implementation Order

1. Verify the existing retrieval baselines and record their actual corpus, query, ground-truth, and metric configurations.
2. Implement deterministic document-level permission assignment and report its realized distributions.
3. Test the policy independently and verify chunk inheritance.
4. Implement pre-filtering and post-filtering with identical evaluation inputs.
5. Record per-query-user outcomes, security violations, returned result counts, and latency.
6. Compute the defined metrics and paired clustered-bootstrap confidence intervals.
7. Run the pre-registered v1/v2 sensitivity analysis.
8. Only after Phase 3 results are validated, implement and evaluate adaptive routing in Phase 6.