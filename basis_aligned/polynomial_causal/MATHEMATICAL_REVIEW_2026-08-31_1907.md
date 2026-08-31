# THREE-HOURLY MATHEMATICAL REVIEW — 2026-08-31 19:07Z

Convention (§2135): CE numbers are damage above the real model; LOWER IS BETTER.

## Fresh facts this review answers to
The evening's rank experiments: QK-8 patterns BREAK squared attention everywhere (motifs +2.06, tail +2.02
vs baselines ~1.90) while QK-16 crosses spectacularly — motif-only +1.6507 (registered, §2282) and the
uniform ALL-QK-16 grammar +1.2673 census / +1.3498 fresh (claim in flight, §2283) with the dictionaries
retired. A sharp threshold between 8 and 16 demands a mathematical explanation.

## Top three mathematical moves

### 1. The Hadamard-rank law for squared attention → EXECUTED (rungs 186/187; claim 188 ahead of them)
Object: each head's pattern pat = (qᵀk)(q₂ᵀk₂)/128² — the HADAMARD PRODUCT of two Gram-form score matrices.
Theorem: rank(A ∘ B) ≤ rank(A)·rank(B). Truncating each projection to rank r caps each factor's rank at r,
so the pattern's effective rank is ~r² — and the full head's pattern rank is bounded by d_head = 128.
Prediction: the critical per-factor rank is r* ≈ √128 ≈ 11.3 — EXACTLY between the observed 8 (64 < 128,
breaks) and 16 (256 ≥ 128, crosses). Falsifiers queued: r=12 must cross (144 ≥ 128), r=10 must degrade
(100 < 128). Assumption that may fail: rotary mixing spreads rank needs unevenly across factor pairs; a
failure pattern localizes there. Consequence beyond reconstruction: a PREDICTIVE sizing rule for every
attention compression in the program (and 25% cheaper patterns at r=12 if the plateau holds).

### 2. MDL accounting of the uniform grammar (record)
If 188 holds, the program's attention story compresses to ONE sentence: every replaced head is
z = pat_r @ v_real with pat_r from four rank-r SVD factors — 12.1M values for 148 heads (~82k/head vs 655k
full), with the class-dictionary grammar, motif alphas, OV residuals, and trajectory-refit machinery all
RETIRED from the config. The description length of the compiled program drops by more than its CE.

### 3. Rotary-aware truncation (design note)
SVD truncation ignores the rotary pairing (adjacent coordinate pairs rotate together); a pair-respecting
truncation could shift r* down. Cheapest test: compare SVD-r12 vs pair-blocked-r12 at one block. Deferred
behind the law test.

## Pruned
Learned patterns (the weights-only SVD is winning; no fit needed); dictionary refinements (retired if 188
holds); gauge/invariant theory (rung 90 still gated); Hankel (superseded by the pattern story).

## Executed
§2282/§2283 written first. Rungs 188 (claim), 186 (r12), 187 (r10) built, preregistered, dryrun-clean,
queued in that order. Registry synced (Pareto collapse registered; uniform grammar pending claim).
