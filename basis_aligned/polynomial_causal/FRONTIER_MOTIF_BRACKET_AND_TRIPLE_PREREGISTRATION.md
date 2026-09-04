# Frontier: bracket the motif optimum, and compose all three adopted scalings — preregistration

Registered 2026-09-04T11:59Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders, so it is none of §2118/§2125/§2131.

## Why

Three rescalings are adopted, and one open limit blocks the third from being quoted as an optimum:

| | § | operation | fresh | in sample |
|---|---|---|---|---|
| TAIL | §2896 | `LW` × 0.25 | −0.2287 | −0.1530 |
| CP | §2902 | `Dk` × 0.50 | −0.1075 | −0.1613 |
| **T+C** | §2904 | both | **−0.3213** | **−0.2899** → frontier **+2.3522**, interaction only **+0.0149** |
| MOTIF | §2905 | `ALPHA` × 1.25 | −0.1605 | −0.1640 |

**§2905's 1.25 was the top of its grid**, so the motif optimum is **not bracketed** and §2905 recorded its number as a **lower bound**.
§2905 also **refuted my stated direction**: every other block wants *less*, the motif heads want *more*, so the mismatch is
**directional per block** and the size of the motif correction above 1.25 is genuinely unknown.

This rung closes both questions in **one pipeline run**: a motif-only grid **{1.25, 1.5, 2.0, 2.5, 3.0}** to bracket the optimum, and
**T+C+motif** arms at {1.25, 1.5, 2.0} to ask whether the third correction adds to the adopted pair.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the T+C arm reproduces §2904.** `|cost(TC) − (−0.3213)| ≤ .01`. *Worked example:* ≈ **.000**; a miss ≥ .03
  (`b_null_the_TC_anchor_fails`) puts an **adopted** number in question and nothing else here is readable.
- **pred_c — the motif optimum is bracketed.** the best motif-only scale is **not** 3.0, the grid edge. *Worked example:* if the
  optimum is near 1.5–2.0 the curve turns inside the grid; if 3.0 still wins (`c_null_the_optimum_is_still_beyond_the_grid`) the motif
  gains want to be **at least tripled**, which would be a large and surprising statement about the motif construction and would need its
  own rung before anything is quoted.
- **pred_d — adding the motif scaling beats T+C.** `cost(TC) − cost(best T+C+motif) ≥ +.05` nats. *Worked example:* §2904 measured the
  T·C interaction at only +0.0149, so if the motif correction is similarly independent the triple reaches ≈ **−0.46** against TC's
  −0.3213, a margin of ≈ **+0.14**; if the motif correction overlaps what T and C already fix, this reads ≈ **.00**
  (`d_null_the_motif_scaling_adds_nothing`) — and the adopted configuration stays T+C.
- **pred_e — the motif-1.25 arm reproduces §2905.** `|cost(m=1.25) − (−0.1605)| ≤ .01`. Second anchor, tying the grid to the section
  that measured it.

## Nulls

- `b_null_the_TC_anchor_fails` (≥ .03): an adopted number fails to reproduce.
- `c_null_the_optimum_is_still_beyond_the_grid`: the motif gains want ≥ 3× and the bracket has to be widened again.
- `d_null_the_motif_scaling_adds_nothing` (≤ 0): the motif correction is already covered by T+C.

**Adoption rule, stated in advance:** a combined improvement beyond §2904's adopted **−0.3213** may be entered **only if pred_a, pred_b
and pred_e all hold** — baseline reproduces and **both** prior numbers re-anchor. If **pred_c fails**, no motif optimum is quoted at all,
regardless of what the combination arms show.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; ten arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_motif_bracket_and_triple_results.json`, read with
`price` in the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` /
`Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
