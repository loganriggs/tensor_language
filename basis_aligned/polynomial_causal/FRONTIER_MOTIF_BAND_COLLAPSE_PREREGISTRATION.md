# Frontier: collapse the motif-attention band to constants — preregistration

Registered 2026-09-04T09:01Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION — first, because it is the rule this rung most depends on

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A collapse **cost**
is `L2_F(collapsed) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED for reading higher L2 as better;
**§2125 STANDS** — the frontier is norm-2304 at 2.6735.

## Why, and why this control

Companion to `frontier_a5_constant_collapse`, which asks the same question of a5 alone. The choice of arms is fixed by an
**independent** receipt — §2834's constant-write census of the REAL model, different data and a different measurement from anything
here:

| band | summed deletion cost | summed cost under constants |
|---|---|---|
| motif a2–a9 | **3.234 nats** | **0.796** |
| tail a10–a17 | **0.157 nats** | 0.161 |

The tail attention components are very nearly inert in document CE (0.010–0.035 nats each), so collapsing them **cannot
discriminate** — a control that is free by construction proves nothing, which is the §2820 admissibility lesson applied to a band
instead of a head. **The control is therefore the band MINUS a5**, not the tail.

a5 alone carries **2.211 of the band's 3.234 nats (68%)**, so the two collapse arms decompose the band: if the whole band collapses
cheaply, the frontier's entire motif-attention stage is a table of eight constant vectors and the "program" gets structurally simpler;
if only a5 does, the price cliff is one component rather than a stage.

Three arms of §312's published norm-selection pipeline: BASELINE; a2–a9 collapsed; a2–a9 except a5 collapsed. The collapse replaces
each layer's fitted `CV` with ten copies of `Y.mean(0)` and empties `LW` — **fitted values only, never control flow**, so the hook still
computes `cur['lab']` and downstream dictionaries see exactly what they saw before. The parent `ops/frontier_fisher8.py` is unmodified,
and the derived file retargets its single `OUT` assignment so no path can clobber §2125's cited receipt.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* the
  same pipeline on the same data reproduces to ≈ **.00–.02**; anything larger means the derivation perturbed the construction and
  **nothing else here is readable.**
- **pred_b — the whole motif band collapses cheaply.** `cost_band ≤ +.10` nats (POSITIVE = WORSE). *Worked example:* inside the
  frontier these dictionaries are ALREADY class-conditional constant tables, so if the ten rows per layer are near-identical the
  collapse costs ≈ **+.01–.05**; if the class structure is load-bearing, ≈ **+.3 or more** and the motif stage genuinely needs its
  classes.
- **pred_c — the band without a5 also collapses.** `cost_band_without_a5 ≤ +.10`. *Worked example:* if collapsibility is a property of
  the whole stage, ≈ **+.01–.05**; if a5 is the only collapsible member, this arm is cheap only because a5 is left alone and the
  contrast with pred_b tells us so. `a5_implied_share = cost_band − cost_band_without_a5` is reported either way.
- **pred_d — the frontier's row spread tracks the real model's census.** Spearman ρ over the eight band layers between the fitted
  dictionary's `1 − min pairwise cosine` among `CV` rows and §2834's `gain_cv` ≥ **.50**. *Worked example:* if the frontier's fitted
  dictionaries inherit the real component's constancy, ρ ≈ **.7**; if the fit imposes its own structure, ρ ≈ **.0** and the real-model
  census does not predict the construction. Registered on the LAYER axis — the invariant index shared by both artefacts, never a
  sample-indexed loading (§2647/§2649).
- **pred_e — the parameter saving is stated exactly**, `10·D + |LINK|·D² − D` per layer and ×8 for the band, D=1152.

## Nulls

- `b_null_the_band_class_structure_is_load_bearing` (`cost_band ≥ +.30`): the motif dictionaries need their classes and the
  simplification does not exist at band scale.
- `c_null_a5_is_the_only_collapsible_one` (`cost_band_without_a5 ≥ +.30`): the result is about one component, not a stage — which
  would still be a real finding, and is registered so it is recognised rather than read as a failure of pred_b.
- `d_null_frontier_spread_is_unrelated_to_the_census` (ρ ≤ .10).

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds**, 0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 3` beside it,
and the ledger's `Price:` line states that explicitly rather than implying a measured forward count. Receipt:
`frontier_motif_band_constant_collapse_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
