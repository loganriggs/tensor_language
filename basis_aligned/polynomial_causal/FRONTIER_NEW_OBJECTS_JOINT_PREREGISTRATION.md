# Frontier: re-optimise around the new objects — rank-32 tail × CP split × motif. Preregistration

Registered 2026-09-04T13:40Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted objects; it neither selects nor reorders.

## Why

§2923 adopted a rank-32 tail projection — **+2.2732 in selection / +2.2953 held out** — but **inherited CP ×0.80 and motif ×1.25 from
§2912**, where both were chosen against a *uniform* tail term. §2923 said so in its own text and registered this as the next step: the
composition shortfall was **0.0275 of a 0.0542 standalone gain**, meaning the terms correct partly the same error and the inherited
constants are the wrong ones for the new tail object.

§2922 then showed the CP correction is itself **not a scalar**: leaving the top 128 units untouched and halving the other 2176 reads
**−0.1280** against uniform's −0.1074, and it transports (−0.1123 vs −0.0908 held out). So the CP axis here is **categorical** — the
incumbent scalar against three splits — rather than a scale sweep.

Three axes against one fitted stack: tail-subspace scale ∈ {0.20, 0.25, 0.30} × CP ∈ {uniform ×0.80, bot128@0.5, bot128@0.8,
bot256@0.5} × motif ∈ {1.15, 1.25, 1.35}. **36 cells, both identity paths controlled in the same run.**

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, nothing else reads.
- **pred_b — both identity paths are physical no-ops.** `|cost(SVD path, all × 1.0)| ≤ .005` **and** `|cost(CP split path, all × 1.0)|
  ≤ .005`. *Worked example:* +0.0001 and 0.0000 in §2921/§2922. **Two different transform paths are combined here for the first time**,
  so both are re-checked in the run that combines them — per the standing rule that a control must travel the measurement's own path.
- **pred_c — the incumbent reproduces §2923.** `|cost(tail r32 s0.25, CP ×0.80, motif 1.25) − (−0.4003)| ≤ .01`. *Worked example:*
  ≈ −0.4003. **Without this the comparison has no reference**, and §2923 is the number to beat.
- **pred_d — re-optimising around the new objects improves on §2923.** `min cell ≤ −0.4003 − 0.01`. *Worked example:* if the inherited
  constants really were mis-matched to the new tail term, the best cell reads ≈ −0.41 to −0.43 and this holds; if §2923's configuration
  is already jointly optimal, the best cell ties −0.4003 and it fails. **A tie is a failure on purpose** — displacing an adopted result
  needs a margin.
- **pred_e — the improvement survives on the held-out window.** `cost_holdout(best) < cost_holdout(§2923 arm)`. *Worked example:* real ⇒
  the ordering holds off the selection window (§2923's did: −0.3757 vs −0.3539); artefact ⇒ it flips. **Mandatory since §2914/§2916**,
  and a strict comparison of two measured quantities, not a ratio.
- **pred_f — the optimum is interior in both continuous axes.** Tail scale interior to {0.20, 0.25, 0.30} and motif interior to
  {1.15, 1.25, 1.35}. **The CP axis is categorical and interiority does not apply to it** — I state that here rather than letting a
  coded check quietly decide it, which is exactly the fault that made §2922's pred_e unsatisfiable. *Worked example:* §2923's optimum
  used 0.25 and 1.25, both interior here; an edge optimum makes the result a **bound** and I will report it as one.

## Nulls

- `b_null_an_identity_path_is_not_faithful` (> .02) — the rung is void.
- `c_null_the_incumbent_fails` (≥ .03).
- `d_null_S2923_is_already_jointly_optimal` — a perfectly good outcome, and the one that closes this line.
- `e_null_the_improvement_is_selection`; `f_null_a_continuous_grid_is_too_narrow`.

**Adoption rule, stated in advance.** The winning configuration replaces §2923 as the frontier of record **only if pred_a, pred_b,
pred_c, pred_d, pred_e and pred_f all hold.** Any one failing and §2923 stands. Whatever the verdict, **both the selection and held-out
numbers are quoted together**, and the selection bias is reported — it has run +0.0044 (§2914), +0.0197 (§2916) and +0.0246 (§2923) as
the search has widened, and this rung searches 36 cells over three axes with two new object types, so a further increase is expected and
must not be presented as noise.

## Price

**1 full frontier pipeline run + 39 arms × 3 windows of forward evaluation, ≤ 800 GPU-seconds** (§2921's 19 arms took 185.0 s and
§2918's 16 took 167.4 s; SVDs are cached after first use and the CP splits are masked rescales of six `Dk` matrices), 0 backwards,
**0 fitted parameters** — every arm rescales objects already fitted. The parent `ops/frontier_fisher8.py` is **unmodified**. Not
forward-instrumented, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_new_objects_joint_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
