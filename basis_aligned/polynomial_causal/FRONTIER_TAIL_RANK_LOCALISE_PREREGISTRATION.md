# Frontier: localise the rank — and supply the control §2919 did not have. Preregistration

Registered 2026-09-04T13:28Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales singular values of already-fitted maps; it neither selects nor reorders.

## Why

§2919 found that scaling **only the top 64 singular directions** of each tail link map reads **−0.2614**, beating the adopted uniform
`×0.25` at −0.2287, while the bottom of the spectrum is actively **harmful** (bot64 +0.0489, bot256 +0.0114, bot576 +0.0005). That
overturns §2917's isotropy **inference** and says the correction is a **low-rank projection, not a scalar**.

**§2919 had a design gap, and closing it is this rung's first job.** Its uniform control ran through `_apply_tail` — a plain multiply —
**not** through the SVD decompose-and-reconstruct path the split arms used. **Nothing in that rung proved the SVD path is faithful**,
and a lossy reconstruction could in principle manufacture the entire effect: if `U diag(s) Vᵀ` did not return `LW`, every split arm
would carry an unmeasured reconstruction error, and the one that happened to cancel error would look best. I registered an identity
control for exactly this reason in §2918's keep-scale rung and **failed to register one here**. **Until these controls pass, §2919's
−0.2614 is not a result**, which is why nothing was adopted from it.

With the controls in place, this localises the rank: **r ∈ {16, 32, 64, 128, 256} × scale ∈ {0.10, 0.25, 0.40}**, with §2919's `top64`
cell as the reproduction anchor and both axes wide enough that its optimum can be interior.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, **nothing else is readable**.
- **pred_b — the SVD path at scale 1.0 is a physical no-op.** `|cost(all directions × 1.0, through the SVD path)| ≤ .005`. *Worked
  example:* **0.0000** if `U diag(s) Vᵀ` reconstructs `LW` faithfully. **This is the control §2919 lacked.** A failure here means
  §2919's finding is an artefact of the reconstruction and the section must be withdrawn rather than annotated — and it is the
  independent check the standing rule demands before a conclusion-flipping correction is published.
- **pred_c — both uniform routes reproduce §2896.** `|cost(uniform, plain multiply) − (−0.2287)| ≤ .01` **and** `|cost(uniform via the
  SVD path) − cost(uniform, plain multiply)| ≤ .01`. *Worked example:* both ≈ −0.2287. The second half is the sharper test: two
  different code paths for the same mathematical operation must agree, which is what makes the split arms comparable to the adopted
  scalar at all.
- **pred_d — `top64` reproduces §2919.** `|cost(r=64, scale 0.25) − (−0.2614)| ≤ .01`. *Worked example:* ≈ −0.2614; a miss ≥ .03 means
  §2919 does not replicate and the rank story stops here.
- **pred_e — the optimum is interior in rank and scale.** *Worked example:* §2919's optimum sat at the **smallest** rank it tested (64
  of {64, 256, 576}), so the true optimum may be lower; this grid goes down to 16 and up to 256 so 64 can be interior. **Registered
  because interiority is my recurring failure — §2907, §2909 and §2917 all failed it on grids I chose too narrow.** A non-interior
  optimum makes the best cost a **bound**, and I will report it as one.

## Nulls

- **`b_null_the_svd_path_is_not_faithful`** (> .02) — §2919 withdrawn, not annotated.
- `c_null_the_uniform_routes_disagree` (> .02).
- `d_null_S2919_does_not_reproduce` (≥ .03).
- `e_null_the_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** All five hold ⇒ **§2917's isotropy sentence is annotated as an inference this
line overturned** (its measurements stand untouched — more ridge really is monotonically worse), the correction is restated as a
rank-r projection, and the follow-up asks whether the CP side is also low-rank and whether the projection composes with §2912's
configuration. pred_b fails ⇒ **§2919 is withdrawn outright** and the isotropy reading survives. pred_d fails ⇒ recorded as a
non-replication, with no rank claim. **Nothing is adopted from this rung**: §2912's configuration remains the frontier of record, and a
low-rank replacement for its tail term needs its own held-out measurement (§2914/§2916) before it can displace anything.

## Price

**1 full frontier pipeline run + 19 arms × 3 windows of forward evaluation, plus 32 one-off SVDs of 1152×1152 matrices, ≤ 700
GPU-seconds** (§2919's 8-arm run took 132.2 s; the SVDs are cached after first use), 0 backwards, **0 fitted parameters** — the arms
rescale singular values of maps already fitted. The parent `ops/frontier_fisher8.py` is **unmodified**. Not forward-instrumented, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_tail_rank_localise_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
