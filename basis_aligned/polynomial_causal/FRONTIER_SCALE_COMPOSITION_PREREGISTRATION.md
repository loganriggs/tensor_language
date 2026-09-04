# Frontier: do the adopted scalings compose? Three blocks, one run — preregistration

Registered 2026-09-04T11:34Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**; **nonadditivity** is `triple − (sum of
singles)`, **POSITIVE = subadditive**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — nothing here selects, reorders or
re-derives, it **rescales already-fitted objects**, so it is none of §2118/§2125/§2131.

## Why

Two frontier improvements are now adopted and a third is measured but not:

| block | section | operation | fresh gain | status |
|---|---|---|---|---|
| **TAIL** | §2896 | link maps `LW` × 0.25 | **−0.2287** | **adopted** (six reproductions, max dev .0003) |
| **CP** | §2902 | CP units `Dk` × 0.50 | **−0.1075** | **adopted** (anchor to §2883 at dev .0001) |
| **FRONT** | §2895 | residual `A` × 0.50 | −0.1648 | **not** adopted — its `tb` anchor failed by 4.95 nats |

They are different blocks, so the obvious question is whether the gains add. **Neither sign is safe to assume from this construction's
record:** §2888 found front × motif **super**additive (+0.3023), §2892 found the motif layers **sub**additive (all three pairs negative),
§2897 found a **one-way** compensation of 4.9479 vs 0.0215, and §2901 found the tail layers coupled in the optimal *amount* rather than
the sign.

**All eight subsets of {T, C, F}** differ only in what `evalM` sees, so they share **one fitted stack**: one pipeline run gives every
single, every pair and the triple — **the complete interaction structure over the three blocks**, which is the Möbius decomposition the
10:28Z mathematical review ranked third, delivered constructively rather than as an attribution exercise.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the tail arm reproduces §2896.** `|cost(T) − (−0.2287)| ≤ .01`. *Worked example:* six prior runs agree to .0003, so ≈
  **.000**; a miss ≥ .03 (`b_null_the_tail_anchor_fails`) would put the **adopted** number in question and nothing else here is readable.
- **pred_c — the CP arm reproduces §2902.** `|cost(C) − (−0.1075)| ≤ .01`. Same reasoning for the second adopted number.
- **pred_d — the combination beats the best single.** `cost(best single) − cost(TCF) ≥ +.05` nats. *Worked example:* if the blocks are
  roughly independent the triple reaches ≈ **−0.50** against the best single's −0.2287, a margin of ≈ **+0.27**; if they overlap heavily
  the triple barely improves on the tail alone and this reads ≈ **.00** (`d_null_the_combination_does_not_help`), which would say the
  three scalings are three views of one correction and only one of them should be kept.
- **pred_e — the three blocks are not additive.** `|triple − Σ singles| ≥ .05`. *Worked example:* Σ singles ≈ −0.501; perfect additivity
  gives a triple of −0.501 and this reads **.00** (`e_null_the_blocks_are_additive`, the outcome that would make the construction
  unusually well-behaved); heavy overlap gives a triple near −0.30 and this reads ≈ **+.20**, subadditive. The sign is reported either
  way, and the three pairwise interactions are recorded so the source of any gap is visible rather than inferred.

## Nulls

- `b_null_the_tail_anchor_fails` / `c_null_the_cp_anchor_fails` (≥ .03): an adopted number fails to reproduce — the most serious
  outcome available here, and registered so it cannot pass unnoticed.
- `d_null_the_combination_does_not_help` (≤ 0): the three scalings are one correction seen three ways.
- `e_null_the_blocks_are_additive` (≤ .02).

**Adoption rule, stated in advance:** the combined improvement may be entered as a result **only if pred_a, pred_b and pred_c all
hold** — baseline reproduces and **both** adopted singles re-anchor. §2895's front-table number stays unadopted regardless, so a triple
that depends on it is reported but not adopted; that restriction is recorded here so it is not decided afterwards.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; eight arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_scale_composition_results.json`, read with `price` in
the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json`
form (§2853, §2858), under a filename no other section cites (§2876).
