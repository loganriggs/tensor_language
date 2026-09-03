# Preregistration (frozen design; GPU rung) — does the low-dim context-summary law generalize beyond MLP0?

Date: 2026-09-03 03:35 UTC
Owner: Claude (parallel probe lane)
Status: frozen falsifiable DESIGN. To be built at quality (reusing rung517's
hash-pinned source-relation machinery, applied at MLP1 instead of MLP0) and
enqueued when it passes a clean dryrun + smoke; NOT rushed. This document is
the falsifiable artifact; the executable follows at quality.

## Question

§2649 (equality-score copy-TASK effect near-rank-1 across implementations) and
the §2652 companion (MLP0's five source-relation effect is rank-1) both say
context is read through a LOW-DIMENSIONAL summary. Is that a LAW that
generalizes to another context-reading MLP, or is it MLP0-specific? MLP1 is
the natural second site (it reads MLP0+attention1 context; the T/C/I/S
machinery of rungs 480–496 already decomposes its input).

## Object and arms (reuse rung517's exact source-relation split, at MLP1)

Apply rung517's five source-relation grouping of attention0's write — NO, at
MLP1 the context comes from attention1; so group ATTENTION1's exact write by
the same five source relations (SELF/PREVIOUS/NEAR/DISTANT_SAME/DISTANT_OTHER),
sum across heads, plus arithmetic remainder, feeding MLP1's deployed
normalization and bilinear map while attention1's direct residual write stays
native. The 32 subsets and Möbius/Shapley machinery are rung517's, retargeted
to attention1→MLP1. Measure the five groups' singleton CE-effect profiles over
the same 192 absolute positions, both corpora, FIT/SELECT.

## Frozen predictions (mirroring the §2652 rank companion, at MLP1)

- **pred_a — exact instrument:** attention1 five-group + remainder reconstruct
  the deployed write at ≤1e-8; full subset reproduces native MLP1; Möbius
  closure ≤1e-10; planted five-factor tables recover; calls/rows/hashes match.
- **pred_b — MLP1 context is ALSO low-rank:** SVD of the 5×192 group CE-effect
  profile matrix has effective rank ≤ 2.5 on both corpora SELECT (the law
  generalizes) — OR the strong null: rank > 2.5 (MLP1 reads context at higher
  rank, so the low-dim summary is MLP0-SPECIFIC, itself a real finding).
- **pred_c — the structure is stable (group-space left vector):** the 5-dim
  group-space top-left singular vector has cosine ≥ .90 FIT→SELECT within each
  corpus AND prose-vs-structured (the §2649-lesson-correct structural axis;
  random-5-vector null |cos|~.45).

## Strong null and interpretation
pred_a false → instrument repair. pred_b rank > 2.5 → the low-dim-context-
summary law is MLP0-specific, not general (a real boundary on the law). pred_c
false → the structure is site-unstable. A pass (rank ≤ 2.5 at MLP1 too) makes
"context is read through a low-dim per-site summary" a two-site LAW with
compiled-program consequences (the reproducible context per MLP is low-rank).

## Literal price
~517's price at MLP1: one source-relation forward pass per subset × 2 corpora
× 2 roles ≈ 4,000–5,000 full forwards, 0 backwards, single phase; per-token CE
+ the 5×192 profiles stored. Zero deployed parameters. Exact count asserted by
the implementation before enqueue.

## Build-quality gate (self-imposed)
Do NOT enqueue until: the attention1 five-group reconstruction closes ≤1e-8 in
a one-batch check; the dryrun exercises the synthetic rank/stability analysis;
the smoke runs the real first batch of BOTH corpora (per my 02:06 ops proposal
— the recurring partial-batch-smoke gap). This gate is the explicit response to
the 517/518/519 instrument-invalid-first pattern.
