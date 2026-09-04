# Frontier: close the decomposition — price the last two `cfgF` blocks and test whether the shares add up

Registered 2026-09-04T09:59Z (the exact string `date -u` returned in its own tool call immediately before this write; six headers this
session were stamped a minute ahead because the clock was read in the same call that composed them). Before the run. Immutable; the
rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93").
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

The arms restore an approximation to the **real component**, which should lower L2, so the quantity is an **error share**:

    error_share(block) = L2_F(baseline) − L2_F(block restored to real)
      POSITIVE = that block contributes that much of the frontier's error
      NEGATIVE = the approximation BEATS the real component in this stack — §2883 measured exactly that for the CP units, at −0.2140

## Why

§2882 and §2883 priced four blocks and left **+1.0978 nats — 41.1% of the published +2.6735 — unattributed**:

| block | share |
|---|---|
| front MLP tables | +1.0045 |
| motif heads (attention 2–9) | +0.3988 |
| tail dictionaries (attention 10–17) | +0.3864 |
| CP units `c4`–`c9` | **−0.2140** |

The evaluated config is `order2 = cfgF + ['a10L'…'a17L']` with `cfgF = ['a0','m0E','a1v','m1','m2E','m3E'] + ['c4'…'c9'] + ['tailE']`,
so **exactly two blocks remain unpriced**: the early-attention entries `a0`/`a1v`, and `tailE`. Pricing them partitions `cfgF`
completely.

That makes the sum of all six shares against the frontier's own +2.6735 a **closure test** — the first check of whether this
decomposition is a partition at all, or whether interaction terms dominate it. §2883 already measured a **+0.2129** additivity drift on
one pair, and §2880 found strong superadditivity *inside* the MLP stage (+3.2104 against an additive 1.4350), so a large closure gap is
a live outcome and is registered as one, not treated as a failure.

The four prior shares are read from the §2882/§2883 receipts **under frozen hashes** rather than retyped, so the closure arithmetic
cannot drift from the numbers it sums. Per §2879 every manipulated entry is a member of `cfgF` by construction, and pred_d re-checks
that as a **measured** predicate.

| arm | change |
|---|---|
| BASELINE | none |
| early attention off | `a0`, `a1v` omitted — those entries run real (the `a10L`–`a17L` refits are excluded from the drop) |
| tailE off | `tailE` omitted — that entry runs real |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the early-attention entries carry a share.** `share_early ≥ +.10` nats. *Worked example:* `a0`/`a1v` approximate the two
  earliest attention layers, which §2834's census makes substantial in the real model (attn0 1.482, attn1 2.426 nats of zero-ablation
  damage), so ≈ **+.3 to +.8**; ≈ **.00** would mean the frontier's early attention is approximated essentially exactly.
- **pred_c — the `tailE` entry carries a share.** `share_tailE ≥ +.10`. *Worked example:* a single fitted entry standing in for the
  late stack, ≈ **+.2**; ≈ **.00** if it is nearly exact, which is `c_null_the_tailE_entry_is_error_free`.
- **pred_d — both arms are connected.** `|share| ≥ .005` for each. *Worked example:* §2879's rule as a measured predicate — a
  disconnected manipulation reads exactly **.0000**, as `fit_attnd` did three times. If either reads .0000 its share must not be
  reported.
- **pred_e — the decomposition closes.** `|Σ(six shares) − L2_F(baseline)| ≤ .60` nats. *Worked example:* if the blocks were
  independent, the six shares would sum to the frontier's total error and the gap would be ≈ **.0**; with the interactions already
  measured, ≈ **.3–.5** is expected. A gap past **1.20** means **interaction terms dominate the construction** and no per-block share —
  including the four already published — can be read as an attribution. That is `e_null_interactions_dominate`, and it is the outcome
  that would most limit §2882 and §2883, so it is registered rather than left implicit.

## Nulls

- `b_null_the_early_entries_are_error_free` (≤ .02), `c_null_the_tailE_entry_is_error_free` (≤ .02).
- `e_null_interactions_dominate` (closure gap ≥ 1.20): **the per-block attribution fails as a partition.** §2880's superadditivity and
  §2883's drift make this a real possibility, and if it fires the four published shares become lower bounds on involvement rather than
  shares — which is how §2883 already hedged them.

## Price

**3 full frontier pipeline runs, ≤ 900 GPU-seconds** (this family measures 279–283 s for three arms; 375–378 s for four), 0 backwards,
0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not zero.
Receipt: `frontier_decomposition_closure_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites
(§2876).
