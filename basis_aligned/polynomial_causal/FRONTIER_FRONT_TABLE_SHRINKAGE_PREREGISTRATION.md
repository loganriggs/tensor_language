# Frontier: shrink the largest error block — a scale sweep on the front MLP tables. Preregistration

Registered 2026-09-04T10:35Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement to the frontier**.
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735. Both `L2_F` (fresh) and `L2_C` (the **fitting** window) are
reported per arm, since §2890 showed the in-sample curve is what distinguishes overfitting from objective mismatch.

## Why this block

§2890 established that the frontier's components are fitted to a **local** objective (per-layer ridge reconstruction) and scored
**end-to-end**, and that the two disagree by a measurable margin. §2893 tested that on the tail link maps and found every scale below 1
improving — best **−0.2287 nats** — but its anchor failed and **it was not adopted**.

This rung applies the idea to the block that matters most. §2883 measured the **front MLP tables at +1.0045 nats — 37.6% of the
published +2.6735, the largest single block anywhere in the construction**, larger than the motif heads (+0.3988) and tail dictionaries
(+0.3864) combined. Each `tableres` entry is a **token table `tb[ids]`** plus a **low-rank quadratic residual `A`**, both ridge-fitted
to local reconstruction. The two are swept independently.

## The anchors are sound this time, and that is the lesson §2893 paid for

§2877 measured `A := 0` at **+0.7536** and `tb := 0` at **+0.6814**, implemented as `torch.zeros_like(...)`. This rung's `a_scale = 0`
and `tb_scale = 0` arms multiply by zero — **the same operation**, not a similar-sounding one. §2893's anchor failed because
`LW := {}` (the hook's loop never runs, so LINK positions keep the class constant) and `LW[k] := 0` (the loop runs and zeroes them) are
genuinely different; **no such gap exists here**, and pred_b/pred_c check it rather than assume it.

Eleven arms differ only in what `evalM` sees, so they share **one fitted stack** (`ops/frontier_evalarms.py`, validated at a baseline
deviation of exactly **0.0** in §2888/§2889): **one pipeline run instead of eleven**.

| knob | arms |
|---|---|
| `a_scale` (quadratic residual) | 0, .25, .5, .75, .9, 1.1 |
| `tb_scale` (token table) | 0, .5, .75, .9 |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the `A := 0` arm reproduces §2877.** `|cost(a_scale=0) − 0.7536| ≤ .05`. *Worked example:* identical operation, so ≈
  **.000**. If it misses by ≥ .10 (`b_null_the_A_anchor_fails`) the sweep is not manipulating what §2877 manipulated and **nothing here
  is usable** — the §2893 discipline, applied in advance.
- **pred_c — the `tb := 0` arm reproduces §2877.** `|cost(tb_scale=0) − 0.6814| ≤ .05`. Same reasoning; second independent anchor.
- **pred_d — some scale below 1 improves.** at least one arm with `a_scale < 1` or `tb_scale < 1` at fresh cost `< 0`. *Worked example:*
  if the local/end-to-end mismatch is general rather than tail-specific, the residual is over-fitted and ≈ **−.05 to −.3** appears
  around `a_scale ≈ .5–.9`; if nothing improves (`d_null_no_scale_improves`) the mismatch is **specific to the tail dictionaries** and
  §2890's account does not generalise to the front — a real and useful bound on my own top-ranked move.
- **pred_e — the arms are connected.** `|cost(a_scale=0)| ≥ .005` and `|cost(tb_scale=0)| ≥ .005`. §2879's rule as a measured
  predicate: a disconnected manipulation reads exactly **.0000**.

## Nulls

- `b_null_the_A_anchor_fails` / `c_null_the_tb_anchor_fails` (deviation ≥ .10): the manipulation is not what §2877 measured; nothing
  adopted.
- `d_null_no_scale_improves`: the objective mismatch is tail-specific, bounding §2890.

**Adoption rule, stated in advance so it is not decided after seeing the number:** any improvement may be entered as a result **only if
pred_a, pred_b, pred_c and pred_e all hold**. §2893's −0.2287 stays unadopted regardless of what happens here.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s for multi-arm fit-once runs), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so —
the count is absent, not zero. Receipt: `frontier_front_table_shrinkage_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
