# Frontier: is the CP halving a truncation correction? Keep-fraction × scale. Preregistration

Registered 2026-09-04T13:18Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this zeroes and rescales already-selected units; it does not reselect or reorder.

## Why

Reading the pipeline to build the ridge-λ rung turned up a fact that reframes half the adopted correction: **the CP entries are not
fitted at all.** `S[f'c{li}'] = ('cp', li, L[keep], Rw[keep], Dw[:,keep], db)` takes the model's **own** `Left`/`Right`/`Down` weights
and sub-selects units. There is no ridge, no least squares, nothing that could be "too large for the end-to-end objective". So §2902's
adopted `Dk × 0.5` **cannot** be the local-fitting artefact §2890/§2905 describe. Whatever it is, it is something else.

`select_units` keeps the top **2304** units by `‖Dw[:,u]‖·‖L[u]‖·‖R[u]‖`, from a bilinear MLP twice that wide. **The frontier keeps
half the units and the adopted correction halves their weights.** That is a hypothesis, and it has a competitor:

| | claim | what the optimum does as fewer units survive |
|---|---|---|
| **H_truncation** | the halving compensates for truncation: with units removed, the survivors at full weight overshoot what the real MLP computes | the optimal scale **slides down** with the keep fraction |
| **H_independent** | 0.5 has nothing to do with the keep count | the row of optima is **flat at 0.5** |

This sweeps **keep fraction × scale** over the retained units — zeroing the lowest-importance survivors by the same formula
`select_units` ranks with — and asks whether `argmin_scale` **moves** with the keep fraction.

**What this is NOT.** §2118 closed K-reduction as a way to make the frontier *cheaper*, and that stays closed: **no cell here is offered
as a better or cheaper frontier and nothing is adopted.** The object is the *relationship* between optimal scale and keep fraction, not
any cell's L2. And per §2897's refit-time/frozen-stack distinction, the downstream components were fitted under all 2304 units, so this
measures the **frozen-stack** relationship; a refit-at-lower-K version is the registered follow-up if the effect exists.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, **nothing else is readable**.
- **pred_b — the identity cell is a physical no-op.** `|cost(keep 1.0, scale 1.0)| ≤ .005`. *Worked example:* **0.0000** exactly — the
  knob at its identity setting must change nothing. **This is the control §2879 taught me to register**: three rungs there measured
  components that were not installed, and only an explicit no-op control caught it.
- **pred_c — the full-keep half-scale cell reproduces §2902.** `|cost(keep 1.0, scale 0.5) − (−0.1074)| ≤ .01`, with −0.1074 **read from
  §2902's receipt at run time, not remembered**. *Worked example:* ≈ −0.1074; a miss ≥ .03 makes the whole keep-fraction row
  incomparable to the adopted result and pred_d unreadable.
- **pred_d — the optimal scale tracks how many units survive.** `best_scale(keep 0.25) < best_scale(keep 1.0)`. *Worked example:*
  H_truncation ⇒ 0.25 (or 0.125) against 0.5, **holds**; H_independent ⇒ 0.5 against 0.5, **fails**. A strict inequality between two
  quantities that are both grid points, so neither operand can change sign — the trap that spoiled §2800 and §2802.
- **pred_e — the scale grid brackets every optimum.** Each row's best scale is interior to {0.125, 0.25, 0.5, 0.75, 1.0}. *Worked
  example:* **this is my recurring failure — §2907, §2909 and the ridge-λ rung all failed interiority on grids I chose too narrow — so
  it is registered as a predicate rather than discovered afterwards.** The grid deliberately extends to 0.125 so that 0.25 is interior.
  A non-interior optimum makes that row's value a **bound**, and I will report it as one.

## Nulls

- `b_null_the_machinery_is_not_a_noop` (|identity| > .02) — a failure here invalidates every other number in the rung.
- `c_null_the_anchor_fails` (≥ .03).
- `d_null_the_optimal_scale_is_independent_of_how_many_units_survive` — H_independent, and a perfectly good outcome: it would say the
  0.5 is **not** a truncation correction and send the search elsewhere.
- `e_null_the_scale_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** pred_d holds with b, c, e ⇒ the CP correction is restated as a **truncation
compensation**, and the follow-up refits at lower K to test it at fit time rather than eval time. pred_d fails ⇒ recorded as a clean
negative: the 0.5 is independent of the keep count, and the next question is whether it tracks something else (layer, unit importance
mass) instead. **Nothing is adopted either way** — §2912's configuration is unaffected by this rung.

## Price

**1 full frontier pipeline run + 16 arms × 3 windows of forward evaluation, ≤ 600 GPU-seconds** (§2914's 2-arm run took 101.5 s and the
ridge rung's 7 arms took ~131 s; the per-arm cost is three short evaluations plus six masked copies of `Dk`), 0 backwards, **0 fitted
parameters** — the arms zero and rescale weights that are already selected. The parent `ops/frontier_fisher8.py` is **unmodified**. Not
forward-instrumented, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_cp_keep_scale_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
