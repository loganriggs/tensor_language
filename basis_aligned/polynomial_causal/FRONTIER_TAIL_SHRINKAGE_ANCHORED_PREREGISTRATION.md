# Frontier: the shrinkage sweep, re-run with an internal anchor — preregistration

Registered 2026-09-04T10:32Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement to the frontier**.
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735. Both `L2_F` (fresh) and `L2_C` (the **fitting** window) are
reported per arm.

## Why

§2893 measured **every** scale below 1 improving the frontier, best **−0.2287 nats at s = 0.25** fresh and **−0.1530 in sample** —
which would be the campaign's **first real improvement to the published +2.6735**. **It was not adopted.** Its registered anchor
failed: the `s = 0` arm read −0.1863 against §2881's **+0.1740** for what the preregistration asserted was the same manipulation — a
deviation of **0.3603**, with opposite signs.

The cause is exact, and it is an error in my preregistration rather than a disagreement between measurements:

```
new = CV[c].clone()
for k in LW:
    sel = c == k
    if sel.any(): new[sel] = x[sel] @ LW[k]
```

§2881 set `LW = {}`, so the loop never runs and LINK-class positions **keep the class constant `CV[c]`**. §2893's `s = 0` set
`LW[k] = 0`, so the loop runs and those positions are **overwritten with zero**. Different operations.

This rung puts **both** zero-arms in one run — `LW := {}` and `LW[k] := 0` — alongside the scale grid, so **the anchor is measured
internally** rather than imported across rungs.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the `LW := {}` arm reproduces §2881.** `|cost(empty) − 0.1740| ≤ .02`. *Worked example:* it is §2881's exact operation, so
  ≈ **.000**. If it still misses (`b_null_the_anchor_still_fails`, ≥ .05) then my diagnosis of §2893 is **wrong**, the two rungs
  disagree for some other reason, and the whole scale line stays unusable — the clause that can refute my own explanation.
- **pred_c — the two zero-arms genuinely differ.** `|cost(empty) − cost(s=0)| ≥ .10`. *Worked example:* §2893 vs §2881 implies a gap of
  ≈ **.36**; if instead the two arms agree to within .02 (`c_null_the_two_zero_arms_are_the_same`) then the `{}`/`0` distinction is not
  the cause and §2893's failure needs a different explanation.
- **pred_d — some scale below 1 improves.** at least one `s < 1` with fresh cost `< 0`. *Worked example:* §2893 measured all of
  .25/.5/.75/.9 negative, best −0.2287; a re-measurement should reproduce that. If nothing improves
  (`d_null_no_scale_improves`), §2893's headline was an artefact of whatever broke its anchor.
- **pred_e — the scale grid reproduces §2893.** `max |cost(s) − §2893's cost(s)| ≤ .02` over the shared scales. *Worked example:* same
  operation, same pipeline, so ≈ **.000**. This makes the rung a *reproduction* of §2893's curve rather than a replacement, which is
  what lets the improvement be adopted if pred_b also holds.

## Nulls

- `b_null_the_anchor_still_fails` (≥ .05): my §2893 diagnosis is wrong and the scale line stays closed.
- `c_null_the_two_zero_arms_are_the_same` (≤ .02): the `{}`/`0` distinction is not the cause.
- `d_null_no_scale_improves`: §2893's −0.2287 does not reproduce.

**Adoption rule, stated in advance so it is not decided after seeing the number:** the improvement may be entered as a result **only if
pred_a, pred_b, pred_d and pred_e all hold**. §2893's own number stays unadopted regardless of what happens here.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (§2893 measured 121.1 s for nine arms; this has seven), 0 backwards, 0 fitted
parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so.
Receipt: `frontier_tail_shrinkage_anchored_results.json`, read with `price` in the same command the ledger section is written from, in
the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section
cites (§2876).
