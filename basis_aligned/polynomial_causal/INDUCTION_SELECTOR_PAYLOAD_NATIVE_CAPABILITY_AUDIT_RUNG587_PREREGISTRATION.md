# Rung 587 preregistration: independent CPU audit of the clean R586 replication

**Frozen:** 2026-09-03 UTC, after the R586 implementation, focused tests, and model-free dry run were frozen, and before either R586 scientific output existed

## Question and authority boundary

R587 asks whether the future R586 files contain the complete evidence and exactly support the scientific decision that they report. It does not load the model, evaluate a prompt, fit anything, change a scientific threshold, rescue a failed gate, or inspect FINAL_TEST/OOD. A held R587 establishes result integrity for the R586 capability screen; it does not by itself identify an induction subcircuit.

The auditor must not import or trust R586 scoring, decision, fixture, validation, or receipt functions. It may reuse the independently frozen R581 low-level reconstruction and bootstrap implementation, pinned by SHA-256, because those functions were written before R580 outcomes and do not depend on R586. R587 wraps those primitives in a new, independently authored R586 result and receipt envelope.

## Exact reconstruction

Starting from the frozen R578 authority rows and only the saved R586 sequence measurements, R587 must independently reconstruct and compare:

1. exactly 108 FIT/SELECT groups, 3,240 rows, and 3,024 unique token sequences;
2. each sequence's group, split, registered answer, token pair, length, final position, logits, log-normalizer, and both target cross-entropies;
3. all 3,240 row records, including base/donor identities, margins, cross-entropies, paired effect, answer-change flag, family, variant, condition, and stable IDs;
4. all 108 four-cell selector-by-payload factorial records and all 432 selected/neutral/contrast condition-effect records;
5. every factorial, interaction, relation-preserving control, selected-match necessity, selected-versus-neutral selectivity, and non-gated contrast summary;
6. all 86 SHA-defined group-cluster bootstrap cells with exactly 2,000 replicates, the frozen lower/upper NumPy quantiles, every draw-matrix hash, every statistic-vector hash, and their combined hash; and
7. the exact three predicates, failed-clause list, conjunction, terminal verdict, and scalar-string `next_step` implied by those independently reconstructed scores.

Numeric comparisons use zero relative tolerance and absolute tolerance `1e-12`. IDs, ordering, memberships, mappings, lists, booleans, strings, field sets, and decisions must match exactly. Both a held scientific result and a complete scientific null are admissible; malformed or incomplete evidence is an audit failure, not a scientific null.

## Result and receipt envelope

The R586 result must be finite standard JSON and have exactly the 28 frozen fields with exact JSON/Python scalar and container types. In particular, `next_step` must be one scalar string in both terminal paths, `rung` must be integer 586, and `elapsed_seconds` must be a finite nonnegative float. Hidden `NaN`/`Infinity`, tuples, missing fields, and extra fields fail closed.

The result must report exactly FIT and SELECT, with FINAL_TEST and OOD absent from both declarations and all raw evidence. It must report exactly 95 model forwards, zero backwards, no weight updates, 3,024 unique sequences, the pinned checkpoint hash, the frozen R586 implementation/test hashes, and the exact R586 input-hash map. The reusable result contract is independently applied to the complete row evidence and pinned R578 authority.

The receipt must have exactly its 15 frozen fields and types. R587 accepts the source pair only when both files exist, both byte streams remain unchanged across a second read, the receipt bytes parse as finite standard JSON, and the receipt's `result_sha256` equals SHA-256 of the exact result bytes. It also binds the result path, R586 implementation, focused test, preregistration, input hashes, checkpoint, terminal decision, scalar `next_step`, split declarations, and 95/0/no-update price. An absent, unpaired, changing, or mismatched source pair fails closed and produces no audit artifact.

## Pre-outcome adversarial dry run

Before R586 runs, the CPU-only dry run must use the full 108-group authority and independently constructed planted evidence for both:

- a held capability result; and
- a scientific-null result that retains complete raw evidence.

Focused tests must cover the 86-cell census and literal SHA bootstrap formula, full reconstruction, strict finite JSON, exact types, FIT/SELECT closure, and tampering by a list-valued `next_step`, a missing group, a non-finite nested number, changed result bytes, changed receipt bytes, and a missing half of the result/receipt pair. Reduced bootstrap replicate counts are allowed only in planted tests and dry runs for CPU cost; the real audit is hard-coded to all 2,000 replicates in every one of the 86 cells.

## Decision and price

R587 returns `held_independent_audit` only if every authority, raw reconstruction, score, bootstrap, envelope, provenance, split, price, terminal-decision, and atomic receipt-binding check holds. Otherwise it returns `failed_independent_audit` with named failures and preserves the independently recomputed scientific verdict when reconstruction was possible.

R587 is `# BQLANE: cpu`. It makes zero model calls, zero backwards, zero updates, opens no protected split, and writes only its own audit or dry-run namespace. It does not queue or run R586.

The requested R587 receipt is the model-free dry-run receipt, which binds the finalized R587 preregistration, implementation, and focused-test hashes before R586 runs. The eventual R587 audit is one immutable JSON artifact that itself records the exact source-result bytes, source-receipt bytes, auditor hashes, and complete bootstrap traces. No separate R587 audit-result receipt is introduced: it would add no independent evidence beyond the audit artifact's source-pair binding and was not part of the requested output namespace.
