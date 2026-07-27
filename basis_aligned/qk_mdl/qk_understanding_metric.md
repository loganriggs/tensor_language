# The understanding metric (U), version 1 — frozen conventions

*(Tick 222. Logan's directive: a number tracking progress on BOTH explaining residuals
and explaining them simply, with idealized extremes and intuition checks to resist
Goodhart. This document freezes U-v1; changes require a version bump and re-scoring of
the full table.)*

## Definition

For a target object O (a circuit route, patched in isolation, route-only as always):

- **C(O)** — causal content: full-audit ΔCE when O is ablated outright. (Measured, not
  assumed.)
- **F(D)** — fidelity of description D: `1 − ΔCE(D)/C(O)`, ΔCE from patching D in for
  O on the full audit. F=1: perfect reproduction; F=0: as bad as ablation; F<0
  possible (worse than ablation — see kernel heads).
- **L(D)** — description length in bits of EVERYTHING D needs beyond the raw input
  tokens: stored tables and codes at their frozen MDL accounting, and — crucially —
  **any referenced model weights at full freight**. This is the anti-Goodhart clause:
  "just run the model" has F=1 but pays the model's bits; pointing at a dense block is
  not understanding, and the metric knows it.
- **S(D)** — simplicity: `log(L_raw/L) / log(L_raw/L_min)`, with L_raw the raw bits of
  O's verbatim implementation and L_min = 10³ bits (a physical-law-sized description).
  S=0: no compression; S=1: law-sized.
- **U(D) = F(D) × S(D)** ∈ (−∞, 1]. One number; the (F, L) pair is always reported
  beside it because U compresses a frontier into a scalar and the pair is the truth.

## Anchors — the idealized extremes

| description | F | S | U | intuition check |
|---|---|---|---|---|
| verbatim weights | 1.00 | 0.00 | **0.00** | copying isn't understanding ✓ |
| "run the whole model" | 1.00 | ≈0 | **≈0** | pointing isn't understanding ✓ |
| random / absent | ≈0 | any | **≈0** | ✓ |
| worse-than-ablation (offset-kernel heads) | <0 | high | **<0** | actively wrong "simple story" is worse than none ✓ |
| a generating law (10³ bits, exact) | 1.00 | 1.00 | **1.00** | the Newton case ✓ |

## The current ledger, scored (route conventions in RESULTS)

Layer-0 pattern route: C = 0.101, L_raw = 7,418 Mbit.
Layer-1 pattern route: C = 2.703, L_raw ≈ 7,418 Mbit (factor-table form).

| # | artifact | F | L | S | **U** | intuition check |
|---|---|---|---|---|---|---|
| 1 | l0 verbatim tables | 1.000 | 7,418 Mb | 0.00 | **0.00** | ✓ baseline |
| 2 | l0 dictionary frontier (§3) | 0.977 | 493 Mb | 0.17 | **0.17** | l0 well understood, but tables are still tables — mid-low feels right |
| 3 | l0 minimal archetype inventories (§5) | ~0.9 (head-wise) | ~54 Mb | 0.31 | **~0.28** | named classes + small: our best l0 object ✓ |
| 4 | l1 static tables (mean-residual form) | 0.981 | ~1,100 Mb | 0.12 | **0.12** | a big learned lookup — modest ✓ |
| 5 | l1 tables compressed by l1 minimal dictionaries | ~0.96 | ~25 Mb | 0.35 | **~0.34** | current best l1 object ✓ |
| 6 | + oracle 16-dim context adapters (§7d) | 0.996 | needs true factors → references model | ≈0 | **≈0** | oracle cheats; metric correctly refuses it ✓ |
| 7 | + generated context, mixed swiglu (§7i–j) | 0.988 | references block-0 (~2,700 Mb with embeddings/weights) | 0.06 | **0.06** | generator still leans on the dense engine ✓ |
| 8 | sliver-16 (§7n) | 0.996 | ~2,700 Mb (block-0 + projections + embeddings) | 0.06 | **0.06** | located, not understood — exactly the L4 gap ✓ |
| 9 | bigram / window lookups (§7l–m) | ≈0.981 (no gain over tables) | +52 Mb | — | **no ΔU** | nulls add bits, no fidelity — metric ignores them ✓ |

**Reading:** the scoreboard says our best objects are the minimal archetype
inventories (U ≈ 0.3) and everything context-dependent is stuck at U ≈ 0.06 because
every working context description still references the dense block-0 weights. LEVEL 4
PROGRESS = MOVING ROWS 7–8 UP: replacing weight-references inside the 16-token window
with small explicit objects, without losing their F. The metric cannot be gamed by
better pointing, only by genuine replacement.

## The work loop (per Logan)

For each residual: (1) identify the structure computing it (done for the window);
(2) hypothesize a simpler representation; (3) fit it; (4) verify F on the ground-truth
audit; (5) score U; (6) check U against intuition before trusting it. First hypothesis
in flight (tick 223): the window function is dominated by PAIRWISE token interactions
(offset-pair bilinear forms) — the next order after the single-token lookups that
scored null. If pairwise fails, the mixer computes ≥3-way interactions, and we climb
the interaction order explicitly, paying bits at each rung and watching U.

## Ledger additions (tick 264, reviewer-2 completion): fold + medical extraction

Scoring the layer-2 symbol fold and the medical explicit pipeline honestly, with the
anti-Goodhart weight-reference clause enforced.

| # | artifact | F | L | S | **U** | intuition check |
|---|---|---|---|---|---|---|
| 10 | l2 symbol fold, C=zero-ablation | 0.955 | ~2,750 Mb (blocks 0-1 + embeddings) | 0.063 | **0.060** | still references the dense engine — same ceiling as rows 7-8 ✓ |
| 10' | l2 symbol fold, C=mean-ablation (honest) | 0.336 | ~2,750 Mb | 0.063 | **0.021** | corrected C halves F: the fold competes only on the small content part ✓ |
| 11 | medical explicit pipeline (PathMNIST, standalone) | 0.811 | 1.86 Mb (192 filters + linear head, NO model ref) | 0.192 | **0.156** | a genuine REPLACEMENT, not a pointer — our highest-U context object ✓ |

**Reading (the reviewer-2 point, quantified):** the bilin18 symbol fold is a
better-fidelity object but U stays at 0.02-0.06 because computing its 384 codes still
references blocks 0-1 wholesale — the metric correctly refuses to reward relocated
pointing, and under the honest mean-ablation content C its F is only 0.34 (most of
"layer-2 pattern" was positional, tick 262). By contrast the medical explicit pipeline
scores U=0.156 — several times higher — precisely because it is a STANDALONE object:
192 extracted pixel-space filters plus a linear head, referencing no model weights, so
its bits are small and honestly its own. This is the metric working as designed: a real
extraction (medical) outscores a located-but-not-replaced bottleneck (bilin18 fold),
and both are dwarfed by the S=1 law anchor. Cross-modal caveat: the medical F uses
above-chance accuracy content, not CE; internally consistent, not directly comparable
to the CE-based bilin18 rows.
