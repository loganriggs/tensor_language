# Frontier: the interaction between the two largest blocks — preregistration

Registered 2026-09-04T10:12Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). An **error share** is
`L2_F(baseline) − L2_F(block restored to real)`, **POSITIVE = that block contributes that much error**. The **interaction** is
`share_both − (share_front + share_motif)`: **NEGATIVE = the two blocks explain overlapping error**, POSITIVE = superadditive.
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2886 priced every member of `cfgF` and found the decomposition **does not close**: six single-block shares sum to **+1.7928** of the
frontier's **+2.6735**, leaving **+0.8807 (32.9%)** unaccounted. That third is not another block — every one is priced — it is
**interaction structure**, already visible as drifts of +0.0270 (§2882) and +0.2129 (§2883), and as outright superadditivity inside the
MLP stage (§2880: +3.2104 against an additive 1.4350).

This rung measures one interaction term directly, between the two largest blocks: the **front MLP tables** (+1.0045, §2883) and the
**motif heads** (+0.3988, §2882). Their singles sum to +1.4033; if the joint restoration recovers materially less, the two are
explaining the same error and the 32.9% gap has a concrete first component.

**It is also the first use of the fit-once/eval-many pattern** (`ops/frontier_evalarms.py`, written after this hour's measurement showed
eval-only rungs overpay 4×). All four arms differ **only** in what is passed to `evalM`, so they are evaluated against one fitted stack:
**one pipeline run instead of four**, ~95 s instead of ~380 s. pred_e is the instrument check on that refactor.

Prior shares are read from the §2882/§2883 receipts **under frozen hashes** rather than retyped.

| arm | change |
|---|---|
| baseline | none |
| front off | `m*` entries omitted — front MLPs run real |
| motif off | `ML := []` — attention 2–9 runs real |
| both off | both |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the interaction term is large.** `|interaction| ≥ .15` nats. *Worked example:* if the front tables and the motif heads
  explain substantially the same error, the joint recovers ≈ **+1.15** against a singles-sum of +1.4033 → interaction ≈ **−.25**; if
  they are independent, joint ≈ **+1.40** and the interaction is ≈ **.00**, which is `b_null_the_two_blocks_are_additive` and would say
  §2886's 32.9% gap lives somewhere other than this pair.
- **pred_c — the joint restoration exceeds either alone.** `share_both ≥ max(share_front, share_motif) + .10`. *Worked example:* a
  sanity bound — restoring more approximations should recover more error, ≈ **+1.1–1.4** against the front's +1.0045. If it reads below
  either single, the arms are inconsistent and the rung is void.
- **pred_d — both arms are connected.** `|share_front| ≥ .005` and `|share_motif| ≥ .005`. §2879's rule as a measured predicate: a
  disconnected manipulation reads exactly **.0000**.
- **pred_e — the fit-once refactor reproduces §2883's baseline.** `|L2_F(baseline) − §2883's baseline| ≤ .001`. *Worked example:* the
  refactor changes only *when* `evalM` is called, not what is fitted, so ≈ **.0000** (§2876 measured the pipeline's resolution as 0.0 at
  four decimals); a deviation ≥ .01 means the saving came at the cost of a **different construction** —
  `e_null_the_refactor_changed_the_construction` — and then neither this rung nor the pattern may be used.

## Nulls

- `b_null_the_two_blocks_are_additive` (|interaction| ≤ .05): this pair is not where §2886's gap lives; the search moves to other pairs.
- `e_null_the_refactor_changed_the_construction` (deviation ≥ .01): **the ops saving is invalid** and `ops/frontier_evalarms.py` must be
  withdrawn. Registered so that a convenient efficiency win cannot quietly corrupt the science it was meant to accelerate.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 94.1 s per run; four arms per-arm would be ~380 s), 0
backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1` beside it, and the ledger's `Price:` line says so — the
count is absent, not zero. Receipt: `frontier_interaction_front_motif_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
