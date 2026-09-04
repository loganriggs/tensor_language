# Frontier: are the motif band's eight constants doing anything, or can the stage be deleted? Preregistration

Registered 2026-09-04T09:19Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2_F(arm) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — the frontier is norm-2304 at
2.6735.

## Why

§2875 measured that the whole motif band a2–a9 collapses to **eight constant vectors for 0.0000 nats**, removing **42,550,272**
parameters (ten class rows plus four full 1152×1152 link maps per layer). It did not ask the next question: **are the eight constants
themselves doing anything?**

That question matters more than it might look, because §312 is a **price-constrained** construction — its config is counted in
components and heads, not parameters. Replacing eight dictionaries by eight vectors saves parameters but keeps eight component slots.
**Deleting** the stage would free the slots, which is the axis the frontier actually optimises and which bears on "tail dictionaries /
coverage credit", one of the three largest standing gaps.

Three arms of §312's published norm-selection pipeline:

| arm | change |
|---|---|
| BASELINE | none |
| band → constant | `CV` ← ten copies of `Y.mean(0)`, `LW` ← `{}` — reproduces §2875 |
| band → zero | `CV` ← 0, `LW` ← `{}` — the stage writes nothing |

**Deletion is implemented as `CV := 0`, not by switching the entry to the parent's `attnz` kind, and that choice is deliberate.**
`attnz` would change which entry is `alist[0]` and therefore which hook computes `cur['lab']` for the downstream dictionaries — that is
**control flow**, and every rung in this family has changed fitted values only. `CV := 0` leaves every control path identical and
changes only what is written. Derived from `ops/frontier_fisher8.py` (§2125 rung 30), which is **unmodified**; the derived file
retargets the parent's single `OUT` so nothing can clobber §2125's cited receipt.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  §2874 and §2875 measured +2.6736 and +2.6735 on this derivation; past .05 and **nothing else here is readable.**
- **pred_b — the constants are doing work.** `separation = cost_zero − cost_const ≥ +.10` nats. *Worked example:* if the eight mean
  writes carry real signal, removing them costs several tenths while replacing them costs nothing, so the separation is ≈ **+.3 to
  +1.0**; if the stage contributes nothing at all, ≈ **.00** — which is `b_null_the_constants_are_not_doing_work` and would mean the
  whole motif-attention stage is removable.
- **pred_c — the constant arm reproduces §2875.** `|cost_const − 0.0000| ≤ .01`. *Worked example:* it is the identical arm on the
  identical pipeline, so ≈ **.0000**; a deviation past .01 means this derivation differs from §2875's and the comparison in pred_b is
  between two different constructions rather than two arms. This is the cross-rung instrument check.
- **pred_d — deleting the stage is expensive.** `cost_zero ≥ +.20` nats. *Worked example:* eight attention layers' worth of writes
  removed should cost a lot, ≈ **+.5 to +2**; if it reads ≈ **+.02** the stage is nearly free to remove outright.
- **pred_e — the parameter and slot accounting is stated**: 42,550,272 parameters saved by the constant arm, eight component slots
  still occupied; zero slots under deletion.

## Nulls

- `b_null_the_constants_are_not_doing_work` (separation ≤ .02): **the motif-attention stage is deletable**, a strictly larger
  simplification than §2875's, and the one this rung exists to be able to find. Registered so that the "disappointing" outcome for
  pred_b is recognised as the bigger result.
- `d_null_the_whole_stage_is_removable` (`cost_zero < +.05`): the same conclusion reached from the absolute side rather than the
  differential one; both are reported.

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds** (§2874 measured 283 s, §2875 279 s for three arms), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not
zero. Receipt: `frontier_band_constant_vs_deleted_results.json`, read with `price` in the same command the ledger section is written
from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
