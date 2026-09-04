# Frontier: the last two untested blocks — `tailE` and the early-attention linear entries. Preregistration

Registered 2026-09-04T12:31Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders.

## Why

Every block of `cfgF` has been priced (§2882/§2883/§2886) and four have been tested against the scale knob: tail (**adopted**), CP
(**adopted**), front tables (overfitted, unadopted), motif heads (bracketed, **91% redundant** with T+C). **Two remain untested:**
`tailE` (**+0.1597**, §2886) and the early-attention `linear` entries (**+0.0574**).

Both have the same structural handle that responded elsewhere. `tailE` is `('tail', Wp, DICT, LIN)` where `LIN` is the linear correction
— the analogue of the tail band's `LW` and the front tables' `A`. The early entries are `('linear', li, W, b)`, so scaling `W` and
keeping the bias is that operation again. **If §2902's broadened claim — that any LOCAL selection criterion leaves end-to-end slack — is
general, both should respond; if neither does, the claim has a boundary and finding it is worth more than another confirmation.**

The rung also asks what §2906 made pressing: **does anything still add on top of the adopted T+C?** The motif block's entire standalone
−0.1604 was 91% absorbed. If these two behave the same way, the frontier's scale corrections are essentially **one two-parameter
object** and block-by-block scaling is finished.

Solo arms plus a full factorial **on top of T+C**, built with `ops/frontier_evalarms.factorial_arms`; ~24 arms in **one pipeline run**.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the T+C arm reproduces §2904.** `|cost(TC) − (−0.3213)| ≤ .01`. *Worked example:* ≈ **.000**; a miss ≥ .03
  (`b_null_the_anchor_fails`) puts an **adopted** number in question and nothing else here is readable.
- **pred_c — at least one new block improves standalone.** some solo arm with fresh cost `< 0`. *Worked example:* if the broadened claim
  is general, the `tailE` linear correction behaves like the tail band's and reads ≈ **−.05 to −.15**; if **neither** responds
  (`c_null_the_mismatch_does_not_reach_these_blocks`) then the effect is confined to the blocks tested so far and the boundary is
  between them — the more informative outcome.
- **pred_d — the new blocks add on top of T+C.** `cost(TC) − cost(best TC+new) ≥ +.01`. *Worked example:* independent corrections would
  give ≈ **+.05–.15**; full redundancy gives ≈ **.00** (`d_null_they_are_redundant_with_TC`), matching the motif block's +0.0149 and
  making the two-parameter reading the right one.
- **pred_e — the best configuration improves in sample.** `cost_fit(best TC+new) < 0`. *Worked example:* the tail and CP corrections
  improve on both windows; the front tables did not (§2895, overfitting). A fresh-only gain **must not be adopted**, which is why this
  is a predicate rather than a remark.

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- `c_null_the_mismatch_does_not_reach_these_blocks`: the boundary of §2902's claim is found.
- `d_null_they_are_redundant_with_TC` (≤ 0): block-by-block scaling is finished at two parameters.

**Adoption rule, stated in advance:** any configuration beyond §2904's adopted T+C may be entered **only if pred_a, pred_b and pred_e all
hold**, and only when pred_d's increment exceeds **+0.05** — the same materiality bar §2906 applied to the motif block and honoured when
it failed at +0.0149.

## Price

**1 full frontier pipeline run, ≤ 500 GPU-seconds** (~24 arms at ~90 s fit + ~3.5 s per arm ≈ 175 s expected), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so.
Receipt: `frontier_remaining_block_scale_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites
(§2876).
