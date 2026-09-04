# Frontier: does the mismatch reach a block that is not ridge-fitted? Scaling the CP units — preregistration

Registered 2026-09-04T11:29Z (the exact string `date -u` returned in its own tool call immediately before this write). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — norm-2304 at 2.6735. This rung does **not** select, reorder or re-derive anything; it **rescales an already-fitted
reconstruction**, so it is none of the CLOSED items (§2118 half-price/K-reduction, §2125 Fisher selection, §2131 c6–c9 reordering).

## Why

§2900 established that the local/end-to-end objective mismatch is **procedural**: scaling still helps after the front token table is
dropped and the quadratic residual is refitted to cover for it (+0.0378 at scale .9). It phrased the mismatch as *"a property of the
ridge fitting procedure itself"*.

**The CP units test that phrasing precisely, because they are not ridge-fitted.** `c4`–`c9` are **norm-selected CP factors** — the
norm-2304 construction §2118/§2125 spent rungs on — not least-squares solutions.

- **If the mismatch is about ridge fitting**, scaling them should **not** help.
- **If it is the more general fact** that every component is chosen by a **local** criterion while the frontier is scored **end-to-end**,
  it should.

Either outcome sharpens §2900's claim, which is why this is worth a rung rather than an assumption.

A `cp` entry is `('cp', li, Lk, Rk, Dk, db)` with forward `((x@Lk.T)*(x@Rk.T))@Dk.T + db`, so multiplying **`Dk`** scales the whole
quadratic reconstruction while leaving the bias — the exact analogue of the `A` scaling §2895/§2900 applied to the front tables.

**§2883 supplies a same-operation anchor**: it measured removing `c4`–`c9` from the evaluated config at an error share of **−0.2140**,
i.e. a **cost of +0.2140** relative to the frontier — note the sign, since restoring the real MLPs there made the frontier **worse**.
That arm is included here.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — the CP-drop arm reproduces §2883.** `|cost(cp dropped) − (+0.2140)| ≤ .02`. *Worked example:* it is §2883's exact
  operation (prefix-drop from `order2`), so ≈ **.000**; a miss ≥ .05 (`b_null_the_anchor_fails`) means the drop path differs from
  §2883's and **nothing else here is readable.**
- **pred_c — scaling the CP units improves the frontier.** at least one scale with fresh cost `< 0`. *Worked example:* if the mismatch
  is about local-vs-global objectives in general, some scale below 1 helps by ≈ **−.02 to −.10**; if it reads **no improvement at any
  scale** (`c_null_the_mismatch_is_confined_to_ridge_fits`), §2900's phrasing is **correct as written** and the mismatch is a property
  of ridge fitting specifically — a genuine narrowing of my own account, registered so it is recognised rather than explained away.
- **pred_d — the arms are connected.** `|cost(cp dropped)| ≥ .005`. §2879's rule as a measured predicate.
- **pred_e — both windows are reported.** `L2_F` (fresh) and `L2_C` (the fitting window) for every arm, since §2890 showed the in-sample
  curve is what distinguishes objective mismatch from overfitting — and the CP units' fitting data is not the same as the tail's.

## Nulls

- `b_null_the_anchor_fails` (≥ .05).
- `c_null_the_mismatch_is_confined_to_ridge_fits` (no scale improves): §2900's "ridge fitting" phrasing is exactly right, and the
  end-to-end refitting move applies to the ridge-fitted blocks only.

**Adoption rule, stated in advance:** any improvement may be entered as a result **only if pred_a, pred_b and pred_d hold** — the
baseline reproduces, the drop arm anchors to §2883, and the arms are connected. §2896's tail adoption is unaffected either way.

## Price

**1 full frontier pipeline run, ≤ 400 GPU-seconds** (this family measures 104–166 s per multi-arm fit-once run; seven arms here),
0 backwards, 0 fitted parameters beyond the pipeline's own. The parent `ops/frontier_fisher8.py` is **unmodified**. It is **not
forward-instrumented**, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`, and the
ledger's `Price:` line says so — the count is absent, not zero. Receipt: `frontier_cp_unit_scale_results.json`, read with `price` in the
same command the ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form
(§2853, §2858), under a filename no other section cites (§2876).
