# Frontier: does the mismatch reach the motif heads? Scaling their per-head gains — preregistration

Registered 2026-09-04T11:38Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales fitted gains; it neither selects nor reorders, so it is none of §2118/§2125/§2131.

## Why

§2902 broadened §2900's claim: the local/end-to-end mismatch is **not confined to ridge fits** but comes from choosing a component by
**any local criterion** while scoring the construction end-to-end. Two blocks now carry adopted rescalings — tail link maps ×0.25
(§2896, **−0.2287**) and **norm-selected** CP units ×0.5 (§2902, **−0.1075**).

**The motif heads are the largest block not yet tested this way** (+0.3988, §2882). Their per-head gains `ALPHA[li] = (ap, asf)` are
fitted as **ratios of inner products** — a local least-squares criterion — so the broadened claim predicts a scale below 1 should help
here too. **If it does not**, §2902's claim is narrower than it states, and the difference between the blocks that respond and those that
do not becomes the thing to explain — which is a more interesting outcome than another confirmation.

**§2882 supplies a same-operation anchor**: it measured `ML := []` (motif heads off, attention 2–9 real) at an error share of
**+0.3988**, i.e. a **cost of −0.3988** relative to the frontier — note the sign, since restoring real attention there **improved** the
frontier. That arm is included.

Seven arms, one fitted stack, **one pipeline run**.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the motif-off arm reproduces §2882.** `|cost(motif off) − (−0.3988)| ≤ .02`. *Worked example:* it is §2882's exact
  operation (`ML := []`), so ≈ **.000**; a miss ≥ .05 (`b_null_the_anchor_fails`) means this rung's motif path differs from §2882's and
  **nothing else here is readable.**
- **pred_c — scaling the motif gains improves the frontier.** at least one scale with fresh cost `< 0`. *Worked example:* if the
  broadened §2902 claim holds, some scale below 1 helps by ≈ **−.02 to −.10**; if **no scale improves**
  (`c_null_the_motif_gains_are_already_optimal`) then a locally-fitted gain that is *already a ratio of inner products against the real
  head output* is apparently close to end-to-end optimal, and the claim needs qualifying by **what** the local criterion targets, not
  merely that there is one.
- **pred_d — the arms are connected.** `|cost(motif off)| ≥ .005`. §2879's rule as a measured predicate.
- **pred_e — both windows are reported.** `L2_F` and the in-sample `L2_C` for every arm, since §2890/§2895 showed the two windows
  distinguish objective mismatch (tail, CP: in-sample gain ≥ fresh) from overfitting (front tables: fresh gain, in-sample loss).

## Nulls

- `b_null_the_anchor_fails` (≥ .05).
- `c_null_the_motif_gains_are_already_optimal` (no scale improves): §2902's broadening is too broad, and the distinguishing feature is
  what the local criterion is fitted *against* rather than its mere locality.

**Adoption rule, stated in advance:** any improvement may be entered as a result **only if pred_a, pred_b and pred_d hold**. §2896's and
§2902's adoptions are unaffected either way.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; seven arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_motif_gain_scale_results.json`, read with `price` in
the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json`
form (§2853, §2858), under a filename no other section cites (§2876).
