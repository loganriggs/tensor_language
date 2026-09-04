# Frontier: where does the published +2.6735 actually come from? Preregistration

Registered 2026-09-04T09:39Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION, and why this rung measures a *share* rather than a *cost*

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93").
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — the frontier is norm-2304 at 2.6735.

The blocks manipulated here are **approximations standing in for real components**: `motif_hooks` replaces attention 2–9's heads with
scaled value copies, and the tail dictionaries replace attention 10–17. Turning one OFF therefore **restores the real component**, which
should **lower** L2. So the quantity is not a deletion cost but an **error share**:

    error_share(block) = L2_F(baseline) − L2_F(block restored to real),   POSITIVE = that block contributes that much of the error.

## Why, after §2879

§2879 forced a reset: §2874/§2875/§2876 manipulated `fit_attnd` dictionaries that are **not in the evaluated configuration**
(`order2 = cfgF + ['a10L'…'a17L']`, `cfgF = ['a0','m0E','a1v','m1','m2E','m3E'] + ['c4'…'c9'] + ['tailE']`), and their 0.0000-nat
results were vacuous. The standing rule adopted there is to verify the manipulated entry is installed **and say so in the section**.

This rung complies by construction: it manipulates **`ML`** — the motif-head layer list passed directly into `evalM` — and **membership
of the `a10L`–`a17L` entries in `order2` itself. Both are the evaluation's own inputs, not fitted objects that may or may not be
installed. pred_d additionally requires both to move the number, as a measured check rather than a code reading.

Nothing has ever costed the **38 motif heads**, the frontier's largest approximated block, and both of the standing largest gaps —
tail dictionaries / coverage credit, and the price cliff at attention 5, which sits inside the motif band — are on this list.

| arm | change |
|---|---|
| BASELINE | none |
| motif heads off | `ML := []` — attention 2–9 runs real |
| tail dictionaries off | `a10L`–`a17L` removed from `order2` — attention 10–17 runs real |
| both off | attention 2–17 runs real |

Derived from `ops/frontier_fisher8.py` (§2125 rung 30), which is **unmodified**; the derived file retargets the parent's single `OUT`,
under a filename no other section cites (§2876).

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  every rung in this family has read +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the motif heads carry a large share.** `share_motif ≥ +.30` nats. *Worked example:* eight layers of attention replaced by
  scaled value copies is the frontier's crudest approximation, so ≈ **+.5 to +1.5**; if it reads ≈ **.00** the motif heads are
  essentially free of error and the frontier's L2 comes from elsewhere entirely — `b_null_the_motif_heads_are_error_free`.
- **pred_c — the tail dictionaries carry a large share.** `share_tail ≥ +.20` nats. *Worked example:* this pipeline prints a
  tail-attention **increment** of +0.3864 for *adding* them, and §2878 attributed +0.2011 of that to their class structure, so the
  error they introduce relative to real attention should be of that order or larger, ≈ **+.3**; ≈ **.00** would mean the tail
  dictionaries reproduce real attention essentially exactly, which §2878's sensitivity makes unlikely but which is registered.
- **pred_d — both blocks are installed and move the number.** `|share_motif| ≥ .01` **and** `|share_tail| ≥ .01`. *Worked example:*
  this is §2879's rule as a measured predicate — an uninstalled block reads exactly **.0000**, as `fit_attnd` did three times. If
  either reads .0000 here, that arm is disconnected and its share must not be reported.
- **pred_e — the two shares are roughly additive.** `|share_motif + share_tail − share_both| ≤ .30` nats. *Worked example:* if the two
  approximations contribute independent error, the joint restoration recovers their sum, drift ≈ **.05**; a large drift means the
  blocks interact — which §2880 already found *inside* the MLP stage (superadditive by 1.78), so it is a live possibility here and is
  reported with its sign rather than assumed away.

## Nulls

- `b_null_the_motif_heads_are_error_free` (share ≤ .05), `c_null_the_tail_dictionaries_are_error_free` (share ≤ .05). Either would
  redirect the whole search for the frontier's error to the MLP/CP side of the construction, which is a useful negative.

## Price

**4 full frontier pipeline runs, ≤ 1,100 GPU-seconds** (this family measures 279–283 s for three arms, so four should land near 370 s),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports
`gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 4` beside it, and the ledger's `Price:` line says so — the
count is absent, not zero. Receipt: `frontier_error_decomposition_results.json`, read with `price` in the same command the ledger
section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858).
