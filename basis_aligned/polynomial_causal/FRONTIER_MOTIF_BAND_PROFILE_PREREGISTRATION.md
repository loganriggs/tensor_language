# Frontier: which motif layer is the construction actually bad at? — preregistration

Registered 2026-09-04T10:14Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). An **error share** is
`L2_F(baseline) − L2_F(layer restored to real)`, **POSITIVE = that layer's motif approximation costs that much**.
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2885 asked whether attention 5 — the "price cliff", a named largest gap for weeks — is where the frontier's motif approximation costs
it. **It is not.** a5's share is **+0.0597** while the registered control a2 costs **+0.1946**: the control is **3.3× worse than the
target**, `c_null_attn5_is_not_special_in_the_band` fired, and the price cliff is closed as a frontier-side target. §2885 named the
replacement lead in one line — **a2, not a5** — and nothing has studied it.

This rung profiles the whole band: baseline plus each of a2–a9 restored to real, giving every motif layer's error share. All nine arms
differ only in the `ML` list passed to `evalM`, so they share **one fitted stack** under the fit-once/eval-many pattern
(`ops/frontier_evalarms.py`): **one pipeline run instead of nine, ~95 s instead of ~850 s**.

**Two of the nine shares are already known independently** (a5 +0.0597, a2 +0.1946 from §2885, read under a frozen hash), which makes
pred_c a real cross-rung reproduction check rather than a formality: the profile must land on values a separate rung measured, or it is
not measuring the same thing.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — a2 dominates the band.** the argmax layer is **a2** and `top / second ≥ 1.5`. *Worked example:* §2885 measured a2 at .1946
  against a5's .0597, so if a2 really is the band's problem layer it leads the profile by a clear margin, ≈ **2–4×**; if some untested
  layer (a3, a8) exceeds it, pred_b fails and **the lead §2885 named is wrong** — which is exactly what a profile is for. If the band is
  flat (`top/second ≤ 1.15`) that is `b_null_the_band_is_flat` and the motif error is diffuse rather than localised.
- **pred_c — a5 and a2 reproduce §2885.** `|share(a5) − .0597| ≤ .01` **and** `|share(a2) − .1946| ≤ .01`. *Worked example:* the same
  manipulation on the same pipeline, so ≈ **.000**; a deviation ≥ **.02** means this rung and §2885 disagree
  (`c_null_the_two_rungs_disagree`) and neither can be trusted until that is resolved.
- **pred_d — most layers are connected.** ≥ **6 of 8** layers have `|share| ≥ .005`. *Worked example:* §2879's rule generalised — a
  disconnected arm reads exactly .0000. Some genuinely inert layers are expected (§2834 puts a7 at 0.062 and a9 at 0.043 nats of
  real-model damage), which is why the bar is 6 of 8 rather than 8 of 8.
- **pred_e — the fit-once refactor reproduces §2885's baseline.** `|L2_F(baseline) − §2885's baseline| ≤ .001`. *Worked example:* the
  refactor changes only *when* `evalM` is called, so ≈ **.0000** (§2876 measured the pipeline's resolution as 0.0 at four decimals). A
  deviation ≥ .01 means the efficiency pattern changed the construction and **both this rung and `ops/frontier_evalarms.py` must be
  withdrawn** — registered so a convenient saving cannot quietly corrupt the science.

## Nulls

- `b_null_the_band_is_flat` (dominance ≤ 1.15): the motif error is spread across layers, so there is no single layer to fix and the
  motif heads must be improved as a block.
- `c_null_the_two_rungs_disagree` (either deviation ≥ .02).

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (94.1 s per run measured this hour; nine arms per-arm would be ~850 s), 0 backwards,
0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_motif_band_profile_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
