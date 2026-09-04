# Frontier: how much do a2, a3 and a4 overlap? — preregistration

Registered 2026-09-04T10:21Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). An **error share** is
`L2_F(baseline) − L2_F(subset restored to real)`, **POSITIVE = that subset's approximation costs that much**.
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

    subadditivity = (sum of singles) − (triple)          POSITIVE = the three explain OVERLAPPING error
    pairwise      = share(pair) − (single_a + single_b)  NEGATIVE = that pair overlaps

## Why

§2889 profiled every motif layer: the frontier's motif error is a **front-of-band** phenomenon — a2 **.1946**, a4 **.1941**, a3
**.1579**, together 73.5% of the band's **.7441** summed share, while a5 (the "price cliff") is **fifth of eight** at .0597. It also
found the layers strongly **subadditive**: eight singles summing to .7441 against §2882's **.3988** for restoring the whole band at once.

**Subadditivity decides what a fix has to look like**, and §2889 could not localise it: improving a2 alone will recover less than its
.1946 if a3 and a4 cover part of the same error. This rung measures the overlap directly, on the three layers that matter, with all
seven non-empty subsets of {a2, a3, a4} — three singles, three pairs, one triple — plus baseline.

Eight arms differ only in the `ML` list passed to `evalM`, so they share **one fitted stack** (`ops/frontier_evalarms.py`, validated at
a baseline deviation of exactly **0.0** in both §2888 and §2889): **one pipeline run instead of eight**.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the front band is subadditive.** `sum(singles) − triple ≥ +.10` nats. *Worked example:* §2889's whole-band figures imply
  heavy overlap (.7441 → .3988), so on the front three, singles ≈ **.547** against a triple of ≈ **.35** gives ≈ **+.20**; if the three
  are independent the triple equals the sum and this reads ≈ **.00**, which is `b_null_the_front_band_is_additive` — and would mean
  each layer can be fixed on its own merits.
- **pred_c — the singles reproduce §2889.** `max |single(l) − §2889's share(l)| ≤ .01` for l ∈ {2,3,4}. *Worked example:* same
  manipulation, same pipeline, and §2889 reproduced §2885 at deviation exactly 0.0, so ≈ **.000**; ≥ **.03** means the two rungs
  disagree (`c_null_this_rung_disagrees_with_S2889`) and neither is usable.
- **pred_d — the arms are connected.** `|single(l)| ≥ .005` for each of the three. §2879's rule as a measured predicate: a disconnected
  manipulation reads exactly **.0000**.
- **pred_e — most pairs are subadditive.** ≥ **2 of 3** pairwise interactions are negative. *Worked example:* if the overlap is a
  general property of adjacent motif layers, all three pairs read ≈ **−.05 to −.15**; if one pair is superadditive while the others
  overlap, the structure is not uniform and a fix must be designed per pair. Registered separately from pred_b because a large triple
  subadditivity could in principle be carried by a single pair.

## Nulls

- `b_null_the_front_band_is_additive` (≤ .02): the three layers explain **independent** error, each is fixable on its own, and §2889's
  whole-band subadditivity must come from the back of the band instead — a useful redirection.
- `c_null_this_rung_disagrees_with_S2889` (any single deviating ≥ .03).

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (§2888 measured 104.4 s for four arms, §2889 121.6 s for nine), 0 backwards, 0
fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 1`, and the ledger's `Price:` line says so — the count is absent, not zero. Receipt:
`frontier_front_band_interaction_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites
(§2876).
