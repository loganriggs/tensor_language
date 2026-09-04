# Frontier: the MLP side's error shares (CP units, front tables) — preregistration

Registered 2026-09-04T09:41Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION, and why this is a *share*

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93").
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

The entries manipulated here are **approximations standing in for real MLPs**, so omitting one from the evaluated config restores the
real component and should **lower** L2:

    error_share(block) = L2_F(baseline) − L2_F(block restored to real),   POSITIVE = that block contributes that much of the error.

## Why

Companion to `frontier_error_decomposition`, which prices the two attention approximations. The evaluated configuration is

    order2 = cfgF + ['a10L'…'a17L'],   cfgF = ['a0','m0E','a1v','m1','m2E','m3E'] + ['c4'…'c9'] + ['tailE']

so the MLP side is two blocks: the **front tables** (`m0E`, `m1`, `m2E`, `m3E`) and the **CP-unit reconstructions** `c4`–`c9`. Between
this rung and its companion the four blocks partition the construction, and the tail refits are explicitly excluded from this rung's
prefix drop so the two do not overlap.

Every entry manipulated here is a member of `cfgF` **by construction** — the §2879 rule — and pred_d requires each to move the number
as a measured check rather than a code reading. §2131 CLOSED c6–c9 **reordering**; this rung does not reorder anything, it prices the
block, which is a different question and is not on the CLOSED list.

| arm | change |
|---|---|
| BASELINE | none |
| CP units off | `c4`–`c9` omitted — MLPs 4–9 run real |
| front tables off | `m*` entries omitted — the front MLPs run real |
| both off | the whole MLP side runs real |

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the CP units carry a large share.** `share_cp ≥ +.20` nats. *Worked example:* the CP reconstruction is the frontier's
  central approximation and the norm-2304 selection that defines it is what §2118/§2125 spent rungs on, so ≈ **+.5 to +1.5**;
  ≈ **.00** would mean the CP units reproduce real MLPs 4–9 essentially exactly, and the frontier's error lives entirely elsewhere —
  `b_null_the_cp_units_are_error_free`.
- **pred_c — the front tables carry a measurable share.** `share_front ≥ +.10`. *Worked example:* four early MLPs approximated by
  token tables plus low-rank residuals, ≈ **+.2 to +.6**; §2880 measured that dropping BOTH halves of those tables outright costs
  +3.2104, but that is a *deletion* cost against the frontier, not an *error share* against the real model, and the two need not be
  close — which is exactly why this is measured rather than inferred from §2880.
- **pred_d — both blocks are installed and move the number.** `|share_cp| ≥ .01` **and** `|share_front| ≥ .01`. *Worked example:*
  §2879's rule as a measured predicate — an uninstalled block reads exactly **.0000**, as `fit_attnd` did three times. If either reads
  .0000 the arm is disconnected and its share must not be reported.
- **pred_e — the two shares are roughly additive.** `|share_cp + share_front − share_both| ≤ .30`. *Worked example:* independent error
  gives drift ≈ **.05**; §2880 found strong superadditivity *inside* the front tables, so interaction here is a live possibility and
  the drift is reported with its sign rather than assumed away.

## Nulls

- `b_null_the_cp_units_are_error_free` (≤ .05) and `c_null_the_front_tables_are_error_free` (≤ .02). Either would move the search for
  the frontier's error decisively onto the attention side, which the companion rung prices.

## Price

**4 full frontier pipeline runs, ≤ 1,100 GPU-seconds** (this family measures 279–283 s for three arms; four should land near 370 s),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 4` beside it, and the ledger's `Price:` line says so — the
count is absent, not zero. Receipt: `frontier_mlp_side_error_share_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a
filename no other section cites (§2876).
