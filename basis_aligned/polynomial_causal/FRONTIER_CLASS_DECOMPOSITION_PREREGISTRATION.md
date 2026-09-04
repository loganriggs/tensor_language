# Frontier: where do the adopted 0.40 nats come from? Decompose by token class. Preregistration

Registered 2026-09-04T13:50Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135), and a per-class **damage** is likewise CE added above the
real model on that class's tokens. A **gain** is damage removed, **POSITIVE = BETTER**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS**.

## Why

**Every frontier number this campaign has produced is one scalar averaged over 30,720 token positions.** §2923's adopted correction is
worth −0.4003 of them. Nobody knows whether that is a broad improvement or a large fix to two token classes, and **nobody has checked
whether it makes any class worse.**

That distinction matters beyond bookkeeping. The campaign's goal is a *predictive, manipulable, editable* program, and a correction that
buys 0.40 nats on average by trading one class against another is a different object from one that helps everywhere — for prediction,
for editing, and for what any certificate could possibly say. §2926 has just shown that standalone and composed behaviour can invert;
this asks the analogous question one level down, **inside** the number the campaign quotes.

The measurement is already in the pipeline, unused: `evalM` computes `F.cross_entropy(..., reduction='none')` per token and discards
everything but the mean, while `cur['clsmap']` labels every position with one of the ten token classes **in the same order**. This
stashes the per-token vector and groups it — for the real model, the uncorrected frontier, and §2923's adopted configuration.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, nothing else reads.
- **pred_b — the decomposition sums to the scalar.** `|SVD identity| ≤ .005` **and** `|Σ_c n_c·damage(c)/N − L2_F(baseline)| ≤ .002`.
  *Worked example:* the weighted mean must reconstruct **+2.6735** exactly up to rounding. **This is the control that makes every other
  number here meaningful** — if a decomposition does not sum to the thing it decomposes, the grouping is misaligned (wrong batch order,
  wrong flattening) and the per-class figures are fiction. It is the arithmetic analogue of the identity arms §2919 taught me to
  register.
- **pred_c — the gain is concentrated in a few classes.** The two classes with the largest **token-weighted** gain hold ≥ **50%** of the
  total gain. *Worked example:* perfectly even ⇒ 2 of 10 classes hold ≈ 20%; concentrated ⇒ 50–80%. Weighted, not per-token-mean, so a
  tiny class with a huge mean cannot dominate — the quantity that matters is nats removed from the corpus.
- **pred_d — no token class is harmed.** `min_c gain(c) ≥ −0.02`. *Worked example:* helps-everywhere ⇒ every class ≥ 0; a trade ⇒ some
  class reads −0.05 or worse and this fails. **A failure here is a substantive finding, not a defect**: it would mean the adopted
  correction is a redistribution, and that belongs on the record next to +2.2732.
- **pred_e — the class gains sum to the measured improvement.** `|Σ_c n_c·gain(c)/N − 0.4003| ≤ .002`. *Worked example:* ≈ 0.4003. The
  second arithmetic control, on the gain rather than the damage, so a sign or alignment error in the difference cannot hide.

## Nulls

- `b_null_the_decomposition_does_not_reconstruct_the_scalar` — the rung is void and I report it as void.
- `c_null_the_gain_is_spread_evenly_over_classes` — a fine outcome: it would say the correction is a broad recalibration.
- **`d_null_the_correction_harms_some_classes`** — the interesting null.
- `e_null_the_gain_decomposition_is_inconsistent`.

**What I will do with each outcome, stated in advance.** pred_c holds ⇒ name the classes and register a rung asking whether the rank-32
subspace is *specific* to them (does the correction's subspace align with those classes' write directions?), which would connect the
scaling programme to the dictionary structure for the first time. pred_d fails ⇒ record the redistribution explicitly, quote it beside
+2.2732 wherever the frontier is reported, and treat "which classes pay" as a first-class question. **Nothing is adopted or withdrawn on
a decomposition**: §2923 stands as the frontier of record regardless — this changes what is *said* about it, not what it is.

## Price

**1 full frontier pipeline run + 3 arms × 3 windows plus one extra real-model pass, ≤ 500 GPU-seconds** (§2926's 39-arm run took 283.3 s;
this has 3), 0 backwards, **0 fitted parameters**. The per-token CE is already computed by the existing code and merely retained. The
parent `ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 1`. Receipt: `frontier_class_decomposition_results.json`, read with `price` in the
same command the ledger section is written from, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other
section cites (§2876).
