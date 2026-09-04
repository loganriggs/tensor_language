# Frontier: the low-rank tail question, re-asked with the fits frozen — preregistration

Registered 2026-09-04T10:02Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2_F(arm) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why, and what was wrong with §2884

§2884 is **void by its own registered null**. It truncated each tail refit's link maps **inside** the sequential refit loop
(`for li in range(10,18)` with `install(order2)` growing as it goes), so truncating layer 10 changed the stack that layer 11 was then
fitted **against**. Its three arms were therefore three differently-fitted constructions, not three perturbations of one. The symptom
was non-monotonicity — **rank 1 read −0.0056 and rank 8 read +0.1351**, so more rank was worse — and `c_null_rank_does_not_matter`
fired. §2884 records the rank-1 figure and does **not** claim it.

This rung applies the identical truncation **after** the refit loop has completed, so **every arm perturbs the same fitted stack**.
Four arms — baseline, rank 64, rank 8, rank 1 — make it a genuine rank sweep, and **pred_b tests monotonicity directly**, which is the
property whose absence voided §2884.

**The stake.** The tail link maps are **42,467,328 parameters** across eight layers, and §2881 measured their full removal at
**+0.1740 nats**. Rank 1 is **73,728 parameters — a 576× compression**. If the cheap reading survives frozen fits it is the largest
simplification the frontier has admitted; if it does not, §2884's rank-1 number was an artifact of the sequential refit and the tail
side closes.

§2884's costs are read from its receipt **under a frozen hash** rather than retyped, so the comparison cannot drift.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — cost is monotone in rank once the fits are frozen.** `cost(rank1) − cost(rank8) ≥ −.01` **and**
  `cost(rank8) − cost(rank64) ≥ −.01`. *Worked example:* with one construction and a pure SVD truncation, more rank cannot be worse, so
  the gaps are ≥ 0, typically ≈ **+.02–.15**; §2884 measured **−0.1407** for the first gap under sequential refits. **If this fails
  again the sequential-refit hypothesis is wrong and the whole low-rank line needs a different explanation** —
  `b_null_still_nonmonotone`.
- **pred_c — rank 64 is essentially free.** `cost(rank64) ≤ +.02`. *Worked example:* 64 of 1152 directions retained is still a 9×
  parameter reduction; if the maps are effectively low-rank this reads ≈ **.00–.01**, and if even rank 64 costs ≈ **+.10** the maps are
  substantively full-rank and the sweep's lower ranks cannot help.
- **pred_d — the arms are connected.** `|cost(rank1)| ≥ .005`. *Worked example:* §2879's rule as a measured predicate — a disconnected
  manipulation reads exactly **.0000**. Rank 1 is the most aggressive arm; if even it reads .0000 the post-loop truncation never
  reached the evaluated entries.
- **pred_e — rank 1 is cheap.** `cost(rank1) ≤ +.05`. *Worked example:* if §2884's rank-1 reading was real rather than an artifact,
  ≈ **.00**, and 42.5M parameters compress 576× at no cost; if it was an artifact of the sequential refit, ≈ **+.15**, close to
  §2881's +.1740 for removing the maps outright — `e_null_rank1_was_an_artifact_of_the_sequential_refit`.

## Nulls

- `b_null_still_nonmonotone` (either gap < −.05): the fix does not restore monotonicity, the sequential-refit hypothesis in §2884 is
  wrong, and **no low-rank number from this family may be reported** until the cause is found.
- `e_null_rank1_was_an_artifact_of_the_sequential_refit` (`cost(rank1) ≥ +.10`): §2884's attractive number does not survive, the tail
  side closes, and that is a clean negative worth the price.

## Price

**4 full frontier pipeline runs, ≤ 1,100 GPU-seconds** (this family measures 375–378 s for four arms; the added SVDs are 32 matrices of
1152×1152 per arm, negligible), 0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**,
so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 4` beside it, and the ledger's `Price:`
line says so — the count is absent, not zero. Receipt: `frontier_tail_link_lowrank_frozen_results.json`, read with `price` in the same
command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form
(§2853, §2858), under a filename no other section cites (§2876).
