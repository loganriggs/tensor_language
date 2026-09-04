# Frontier: dump the fitted stack so the certificate line can proceed on CPU — preregistration

Registered 2026-09-04T12:34Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93").
§2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS**. **Nothing here changes the construction — it writes it down.**

## Why

The explained fraction is **5.348% / 10.923% / 4.727 nat / 0 of 68**, and the last component is a **certificate count** — the ledger
records "0 of 68 stands; this is a distinct effect-variance metric, proposed as coverage credit". **It has been exactly zero for the
entire campaign.**

Nothing this lane has produced carries an *a priori* bound, including the two adopted rescalings that took the frontier from **+2.6735
to +2.3522**. Those improve the number and **certify nothing**. That is the honest strategic position: the scaling programme is a
manipulability result, not an explanation or a certification one.

The 10:28Z mathematical review ranked **balanced truncation with the Glover bound** second precisely because it would supply the
ledger's first certificate — for a linear system, `‖G − G_r‖_∞ ≤ 2 Σ_{i>r} σ_i` in the Hankel singular values, an **a priori** bound
rather than a measured one. Its stated blocker was mundane: **the fitted matrices are never written to disk**, so every spectrum
computation would need a GPU pipeline run.

This rung removes that blocker permanently. It fits the published stack once and **persists it** with
`ops/frontier_fitcache.save_stack`, after which Gramians, Hankel spectra and per-matrix spectra can all be computed on CPU at **zero GPU
cost**.

**A note on the tool, recorded because I downgraded it myself.** `frontier_fitcache.py` was written at 11:06Z to save GPU time, and the
12:07Z ops review downgraded it when the GPU turned out to be 68% idle — saving fit time no longer bought wall-clock. Its real value is
different from the one it was built for: **getting the fitted objects onto disk.** Better to say that than to quietly reuse it.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and the stack being dumped is not the published one.
- **pred_b — the saved stack round-trips exactly.** `verify_stack(in-memory, reloaded)` reports ok with **max deviation 0.0**. *Worked
  example:* `torch.save`/`torch.load` of float tensors is exact, so ≈ **0.0**. The verification is **recursive over tuples and dicts**
  because `S` maps a key to a tuple containing a dict of tensors — a top-level comparison would be structurally incapable of seeing a
  changed link map, which is exactly how `ops/fastload.py` shipped broken at 06:24 reporting "bit-identical over 218 tensors".
- **pred_c — the dump contains the expected objects.** ≥ **20** tensors compared, and the entry-kind manifest is recorded (`attnd`,
  `tableres`, `cp`, `tail`, `linear`). *Worked example:* the evaluated config has ~28 entries; a dump with 3 tensors would be
  well-formed and useless.
- **pred_d — the file is written**, with its size in MB recorded so the CPU-side work knows what it is loading.
- **pred_e — a stack RELOADED from disk reproduces L2.** `|L2_F(reload arm) − L2_F(baseline)| ≤ .001`. *Worked example:* the reload arm
  **replaces every fitted entry with the one read back from disk** before evaluating, so ≈ **.000**; a deviation ≥ .01
  (`e_null_the_reloaded_stack_does_not_reproduce`) means the persisted object is not usable and the certificate line stays blocked.
  **This is the only predicate that tests usability rather than well-formedness**, and the reload arm was rewritten during construction
  precisely because the first version would have evaluated the in-memory stack and tested nothing.

## Nulls

- `b_null_the_dump_is_not_faithful` (round-trip mismatch of any size).
- `e_null_the_reloaded_stack_does_not_reproduce` (≥ .01): the dump is well-formed but unusable, and the certificate line stays blocked
  on the same mundane obstacle.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds**, 0 backwards, 0 fitted parameters beyond the pipeline's own, plus a one-off disk
write of the fitted stack (expected order 100–500 MB, recorded in the receipt). The parent `ops/frontier_fisher8.py` is **unmodified**.
It is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`.
Receipt: `frontier_stack_dump_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename no other section cites (§2876).
