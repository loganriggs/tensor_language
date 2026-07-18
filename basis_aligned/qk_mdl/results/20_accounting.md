# The consolidated MDL accounting: every description layer, priced and verified

The directive's "we can in fact measure" deliverable: every reduction the program has
built, with its structural bits (32-bit convention), estimation data, audited ΔCE, and
what *verification beyond ΔCE* it carries. bilin18 (546M params ≈ 17.5 Gbit raw), audits
at T=512.

| description layer | structural cost | est. tokens | ΔCE | verification beyond ΔCE |
|---|---|---|---|---|
| live model (reference) | 546M floats | — | 0 | — |
| L0 folded factors (exact) | tables ≡ weights | 0 | ~1e-15 | fp64 gates |
| L0 grand codebook, trained (results/09) | 9.9M floats | 2.1M train | **−0.019** | KL-faithfulness split |
| windowed-D, raw tables, W=6 (results/11) | 2.1B floats (raw) | 524k | +0.059 | cross-region audit |
| ...tables at vq1024 (free) | 44M fl + 19M idx bits | 524k | ≈same | — |
| **champion combo tables, W=6** (this run) | **2.5M fl + 18M idx bits** | 524k | **+0.042** | denoising bonus replicates |
| combo tables, W=4 (results/16) | 2.5M fl + 18M bits | 524k | +0.089 | cross-region +0.089 |
| rulebook, any single layer (results/19) | ~32k bits/layer·head set | audit slice | ≤+0.008 | — |
| rulebooks, ALL layers composed (this run) | ≈0.66 MB total | audit slice | **+0.190** | TS-1: 6× superadditive |
| **TOTAL SYSTEM (tables + rulebooks)** | ~12.7 MB total | 524k | **+0.256** | TS-2: cross-family ≈ additive |
| contextual core atoms (results/12,18) | 2 heads + ~4–16 gains | — | (kept live) | null-calibrated monosemanticity; 3 cards with set-ablations |

## The three total-system findings

**TS-1 — per-layer freedom ≠ stack freedom, again.** The 3%-density rulebooks cost
≤+0.008 at any single layer but +0.190 across all 18 — the composition law holds within
the rulebook family (6× superadditive). The whole-model routing claim must be quoted at
+0.19, not "free."

**TS-2 — the first cross-family additivity.** Tables (+0.042) and rulebooks (+0.190)
together cost +0.256 — interaction only +0.024. The two reductions consume different
error budgets: tables coarsen long-range *content*, rulebooks prune weak *interactions*.
(Within-family composition remains superadditive; across these families it is not.)

**TS-3 — denoising compounds.** The champion combo tables (r=32 basis + vq1024
coefficients) audit at +0.042 at W=6 — *better* than the raw 2.1B-float tables (+0.059).
Low-rank truncation's noise removal survives full composition.

## The sentence the accounting supports

A 546M-parameter transformer's computation, to +0.26 nats (or +0.042 keeping full
patterns): **~12.7 MB of readable structure** — token-class tables for all long-range
content, class-interaction rulebooks for all attention routing — plus a 6-layer live
window whose irreducible contextual core is two named heads and a handful of named
gains, each carrying falsifiable causal verification.

Files: `../combined_final.py/.json`; components in results/11, 16, 18, 19.
