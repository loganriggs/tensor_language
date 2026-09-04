# Frontier: does the rank-32 projection compose? The adoption candidate. Preregistration

Registered 2026-09-04T13:34Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales singular values of already-fitted maps; it neither selects nor reorders.

## Why

§2921 established, with its controls passing (`svd_identity` **+0.0001**, both uniform routes agreeing to **0.0000**, §2919 replicating
to **0.0000**), that the tail correction is a **rank-32 projection**: shrinking 32 of 1152 singular directions reads **−0.2828** against
the adopted uniform scalar's −0.2287, **interior in both axes**, and **−0.2872 held out**.

**That was the tail term alone.** §2912's adopted frontier is a *composition* — tail ×0.30 uniform, CP ×0.80, motif ×1.25, **+2.2999 in
selection / +2.3171 held out** — and §2904 showed these terms **interact** (tail × CP by +0.0149, and the three-way non-additivity was
0.2127 nats). **A standalone gain of 0.0541 does not entitle anything to displace an adopted composed result.**

This swaps **only** the tail term, holding CP and motif at §2912's adopted values, and measures the whole configuration on both windows.
It is the adoption candidate, and it is built so it can fail.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, nothing else reads.
- **pred_b — the SVD path at scale 1.0 is a physical no-op.** `|cost(all × 1.0 through the SVD path)| ≤ .005`. *Worked example:*
  +0.0001 in §2921. Carried forward because the measurement path is the same one, and a control is only worth what it costs to repeat.
- **pred_c — the adopted configuration reproduces §2912.** `|cost(tail .30, CP .80, motif 1.25) − (−0.3736)| ≤ .01`. *Worked example:*
  ≈ −0.3736, reproduced twice already (§2909, §2912 at deviation 0.0000). **Without this the comparison has no reference.**
- **pred_d — the rank projection alone reproduces §2921.** `|cost(rank 32, scale 0.25, no CP/motif) − (−0.2828)| ≤ .01`. *Worked
  example:* ≈ −0.2828. Together with pred_c this pins both endpoints of the swap in the same run, so the composed number cannot be an
  artefact of drift in either.
- **pred_e — the rank projection improves the composed configuration.** `min cell ≤ cost(G_uniform) − 0.01`. *Worked example:* if the
  0.0541 standalone gain survives composition even partly, the best cell reads ≈ −0.40 to −0.43 against −0.3736 and this holds; if the
  tail term's extra precision is redundant with what CP and motif already fix, the best cell ties −0.3736 and it fails. **A tie is a
  failure here on purpose** — displacing an adopted result requires a margin, not parity.
- **pred_f — the improvement survives on the held-out window.** `cost_holdout(best cell) < cost_holdout(G_uniform)`. *Worked example:*
  real ⇒ the held-out ordering matches the selection ordering (§2921's projection was *better* held out, −0.2872 vs −0.2828); selection
  artefact ⇒ the ordering flips off the window that chose it. **§2914/§2916 made this mandatory before any number is quoted**, and it is
  a strict comparison between two measured quantities, not a ratio.
- **pred_g — the optimum is interior in rank and scale.** *Worked example:* §2921's optimum was rank 32 with {16, 32, 64} available here
  and scale 0.25 inside {0.10, 0.25, 0.40}. **Registered because interiority is my recurring failure — §2907, §2909, §2917 and §2919 all
  failed or bounded on grids I chose too narrow.** A non-interior optimum makes the adoption number a **bound**, and I will say so.

## Nulls

- `b_null_the_svd_path_is_not_faithful` (> .02) — the rung is void.
- `c_null_the_adopted_configuration_fails` (≥ .03); `d_null_S2921_does_not_reproduce` (≥ .03).
- `e_null_the_projection_adds_nothing_in_composition` — the tail's extra precision is redundant with CP and motif.
- `f_null_the_improvement_is_selection`.
- `g_null_the_grid_is_too_narrow`.

**Adoption rule, stated in advance.** The rank-32 configuration replaces §2912 as the frontier of record **only if pred_a, pred_b,
pred_c, pred_d, pred_e, pred_f and pred_g all hold** — every anchor, the composed margin, the held-out survival and interiority. Any
one failing and §2912 stands. **The receipt also reports the composition shortfall** (`standalone gain − composed gain`): §2904 measured
0.2127 nats of three-way non-additivity, so some shortfall is expected and its size is the interesting quantity regardless of the
verdict.

## Price

**1 full frontier pipeline run + 13 arms × 3 windows of forward evaluation, ≤ 600 GPU-seconds** (§2921's 19-arm run took 185.0 s; SVDs
are cached after first use), 0 backwards, **0 fitted parameters** — every arm rescales singular values of maps already fitted, with CP
and motif held at constants read from §2912. The parent `ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_rank_composition_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
