# Frontier: does the ADOPTED §2912 optimum transport, and did the 95-cell search buy anything real? Preregistration

Registered 2026-09-04T13:08Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A **gain** is
`L2(baseline) − L2(arm)`, **POSITIVE = BETTER**. §2128/§2129/§2133/§2134 RETRACTED; §2125 STANDS.

## Why

§2912 adopted **tail .30 / CP .80 / motif 1.25 → L2_F +2.2999** under a rule registered in advance whose predicates all held, and the
section recorded its own qualification in the same breath: **all 95 cells of that three-axis search were scored on the same fixed 120
documents.** Its sibling `frontier_scale_holdout` transports the §2904 **two-parameter** configuration, which is a **lower bound** on
the bias here — two single-axis rungs are far less selected than a 95-cell grid.

This rung measures the quantity that matters for the campaign's headline: **baseline, §2904 TC, and the §2912 adopted config, each on
the selection window and on a held-out window**, in one run. Windows are built by the identical scan continued to 240 rows, so
`rows[:120]` is bit-identical to the FR every prior rung used and `rows[120:240]` chose nothing.

Two derived quantities come out that nothing else can produce:

| quantity | definition | reads on |
|---|---|---|
| selection bias | `gain(FR) − gain(FR2)` per config | how much each number flatters itself |
| **EXTRA** | `gain(G) − gain(TC)` per window | **the 95-cell search bought 0.0523 nats on FR; is any of it there off that window?** |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, **nothing else is readable**.
- **pred_b — the adopted gain reproduces on the selection window.** `gain_FR(G) ≥ 0.25`. *Worked example:* §2912 measured **+0.3736**.
  **This exists to pin the sign of pred_c's reference quantity** (standing rule: never `X ≥ k·Y` with `Y` possibly ≤ 0); pred_c is coded
  to require pred_b first.
- **pred_c — at least three quarters of the adopted gain transports.** `gain_FR2(G) ≥ 0.75 × gain_FR(G)`. *Worked example:* real ⇒
  ≈ +0.37 against a threshold ≈ +0.28, holds; pure selection ⇒ ≈ 0.00, fails outright.
- **pred_d — the held-out window is a comparable comparator.** `|L2_F2(baseline) − L2_F(baseline)| ≤ 0.50`. *Worked example:* two draws
  from one scan sit within a couple of tenths; a bigger offset means the windows are not like for like and **pred_c is uninterpretable**.
- **pred_e — the adopted gain does not change sign off the selection window.** `gain_FR2(G) > 0`. *Worked example:* the weakest
  transport claim; failing it with pred_a and pred_d holding is a serious finding against §2912.
- **pred_f — the extra search beyond TC survives off the selection window.** `EXTRA(FR2) = gain_FR2(G) − gain_FR2(TC) > 0`.
  *Worked example:* on FR, `EXTRA = 0.3736 − 0.3213 = +0.0523`. If the three-axis grid found real structure, FR2 shows a positive
  EXTRA of similar order (≈ +0.03 to +0.05). **If EXTRA(FR2) ≈ 0.00 or negative, the 95-cell search was fitting documents and the honest
  frontier is §2904's two-parameter object** — an outcome that **costs me my own most recent adoption**, which is exactly why it is
  registered before the numbers are seen.

## Nulls

- `b_null_the_adopted_gain_does_not_reproduce` (< 0.25).
- `c_null_the_gain_is_an_artefact_of_the_selection_window` (`gain_FR2 ≤ 0.5 × gain_FR`).
- `e_null_the_correction_is_window_specific` (`gain_FR2 ≤ 0`).
- `f_null_the_95_cell_search_bought_nothing_transportable` (`EXTRA(FR2) ≤ 0`).

**What I will do with each outcome, stated in advance.** pred_c and pred_f hold ⇒ §2912's +2.2999 stands, with a measured bias beside
it. pred_e holds but **pred_f fails** ⇒ **§2912 is restated on the record: the adopted configuration reverts to §2904's two parameters
and the three-axis grid is closed as selection**, since a search that buys nothing off its own window is not a finding. pred_e fails
with a and d holding ⇒ a negative result against §2896/§2902/§2904/§2912 together, and per the standing rule a conclusion-flipping
correction requires an independent physical control before anything is withdrawn. **No outcome licenses searching for better scalars on
FR2** — that would repeat the exact error under test.

Re-measuring TC here also **cross-checks `frontier_scale_holdout`** on an independently-run pipeline; a disagreement beyond the ~0.003
CUDA-atomics wobble is itself reportable.

## Price

**1 full frontier pipeline run + 4 extra forward evaluations, ≤ 450 GPU-seconds** (§2912's 27-arm run took 184.4 s; this has 3 arms
across 2 windows plus one extra `classify2` pass), 0 backwards, **0 fitted parameters** — both configurations are frozen constants read
from receipts. The parent `ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt: `frontier_holdout_adopted_results.json`, read
with `price` in the same command the ledger section is written from, in the canonical `Price:` / `Results:` form (§2853, §2858), under a
filename no other section cites (§2876).
