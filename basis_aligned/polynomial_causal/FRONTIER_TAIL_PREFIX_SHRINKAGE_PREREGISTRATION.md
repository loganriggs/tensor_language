# Frontier: is the tail band a cascade? Prefix shrinkage — preregistration

Registered 2026-09-04T11:10Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — norm-2304 at 2.6735.

## Why

§2899 found that per-layer tuning **does not compose**. Every one of the eight tail layers prefers a scale of **.05–.10** when tuned
alone and gains more that way than at .25 — but composing all eight optima gives **−0.2092**, **worse** than the global 0.20's
**−0.2290**, with the additive prediction (−0.2906) overshooting by **.0814**. §2899 closed with a hypothesis it did not test: the tail
dictionaries form a **cascade** — a10L's output is part of a11L's input — so individually optimal shrinkage **compounds down the chain**
into collective over-shrinkage.

That hypothesis has a sharp and cheap consequence. Shrink a **prefix** of the band at the common per-layer optimum (.05) and lengthen it
one layer at a time. **Independent layers** would give a cost falling monotonically with prefix length; a **cascade** gives a gain that
**saturates or reverses** once enough of the chain is shrunk.

Ten arms differ only in what `evalM` sees, so they share **one fitted stack**: **one pipeline run**.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the prefix gain saturates or reverses.** the best prefix length is **< 8**. *Worked example:* if shrinkage compounds down
  the cascade, the curve bottoms around **n = 3–5** and then rises, so the argmin is interior; if the layers are independent every added
  layer helps and the argmin is **8** (`b_null_each_added_layer_helps`), which would refute §2899's cascade hypothesis and leave the
  non-composition unexplained.
- **pred_c — the global 0.20 arm reproduces §2898.** `|cost − (−0.2291)| ≤ .01`. *Worked example:* ≈ **.000**; the **seventh**
  reproduction of the adopted effect, and it keeps this curve comparable to every earlier one.
- **pred_d — the single-layer arm reproduces §2899.** `|cost(prefix=1) − (−0.0847)| ≤ .01`. *Worked example:* prefix length 1 is a10L
  alone at .05, exactly §2899's per-layer arm, so ≈ **.000**. Together with pred_c this ties the curve to independently measured numbers
  **at both ends**.
- **pred_e — the arms are connected.** `|cost(prefix=1)| ≥ .005`. §2879's rule as a measured predicate.

## Nulls

- `b_null_each_added_layer_helps` (argmin = 8): the cascade account is wrong and §2899's over-shrinkage needs a different explanation.
- `c_null_the_global_anchor_fails` (≥ .03).

**Adoption note:** diagnostic. **Nothing here is adopted** — §2896's global scale stands as the best known configuration, and §2899
already showed the per-layer alternative is worse.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; ten arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_tail_prefix_shrinkage_results.json`, read with
`price` in the same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` /
`Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
