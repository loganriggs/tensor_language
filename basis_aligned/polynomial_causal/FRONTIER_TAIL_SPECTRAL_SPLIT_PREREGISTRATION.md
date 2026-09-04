# Frontier: is the excess really isotropic? Scale half the spectrum instead of all of it. Preregistration

Registered 2026-09-04T13:23Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales singular values of already-fitted maps; it neither selects nor reorders.

## Why

§2917 refuted the under-regularisation hypothesis decisively — more ridge is monotonically **worse** at every level tested (+0.0195
through +0.0972) while uniform `×0.25` is worth **−0.2287** — and read that as: *the end-to-end excess is isotropic, not concentrated in
the low-eigenvalue directions ridge attacks.*

**That reading is an inference from one failed family, not a measurement.** Ridge is one particular anisotropic shrinkage; its failure
rules ridge out and *hints* at isotropy without demonstrating it. §2917's section states the inference as the finding, so it is my job
to test it before it hardens into an assumption.

This measures it directly. Each tail link map is `LW = U diag(s) Vᵀ`. Applying the **same adopted scalar 0.25** to only the **top r**
singular values, or only the **bottom** ones, holds the shrinkage-per-direction fixed and varies only **which** directions receive it.
Three ranks × two halves — r ∈ {64, 256, 576}, and 576 is exactly half of 1152 so `top` and `bot` are equal-sized there — against the
uniform arm measured in the same run.

| | prediction |
|---|---|
| **isotropic excess** | no split beats uniform; **both** halves carry gain; the two halves roughly **add** to the whole |
| **concentrated excess** | some split **beats** uniform — the gain lives in a subspace, and the correction is not a scalar at all but a **projection** |

**Either answer advances on §2917's inference, and the concentrated one would matter more**: it would say the two adopted parameters are
a coarse stand-in for something with structure, and point at exactly which subspace.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, **nothing else is readable**.
- **pred_b — uniform scaling reproduces §2896.** `|cost(uniform) − (−0.2287)| ≤ .01`. *Worked example:* ≈ −0.2287, now reproduced six
  times. This is the in-run control every split is compared against; a miss ≥ .03 makes pred_c unreadable.
- **pred_c — no spectral split beats uniform shrinkage.** `min_splits cost ≥ cost(uniform) − 0.01`. *Worked example:* isotropic ⇒ the
  best split lands **above** −0.2287 (each half alone recovers only part of it, say −0.08 to −0.15) and this holds; concentrated ⇒ some
  split reads **−0.26 or better**, beating uniform with half the shrinkage, and this fails. **The bar is one-sided on purpose**: the
  question is whether anisotropy *helps*, and a split that merely ties uniform is not evidence of structure.
- **pred_d — both halves of the spectrum carry gain.** `cost(top r) < 0` **and** `cost(bot r) < 0` at every r. *Worked example:*
  isotropic ⇒ both negative at all three ranks; if the excess lived only in the top 64 directions, `bot64` would read ≈ 0.0000 and this
  fails. Two independent quantities, each compared to zero — no ratio, no sign-flipping denominator (the §2800/§2802 trap).
- **pred_e — the two halves add up to the whole.** `|cost(top 256) + cost(bot 256) − cost(uniform)| ≤ 0.05`. *Worked example:*
  isotropic and near-linear ⇒ the halves sum to ≈ −0.2287 and the residual is ≈ 0.00; strong interaction ⇒ a residual like §2904's
  **0.2127** non-additivity, which would say the directions are not independently correctable. **Informative either way, and it is a
  difference rather than a ratio.**

## Nulls

- `b_null_the_anchor_fails` (≥ .03).
- **`c_null_the_excess_is_concentrated_in_a_subspace`** — the interesting one: a split beats uniform, and the correction has structure.
- `d_null_one_half_of_the_spectrum_carries_nothing`.
- `e_null_the_spectral_halves_interact`.

**What I will do with each outcome, stated in advance.** c, d and e all hold ⇒ §2917's isotropy reading is **upgraded from inference to
measurement** and recorded as such, and the scalar correction is the right object. pred_c fails ⇒ the finding is a **subspace**, the
follow-up localises it by sweeping r finely on the winning half, and §2917's section is annotated to say its isotropy sentence was an
inference this rung overturned — **with an independent physical control before anything is withdrawn**, per the standing rule for a
conclusion-flipping correction. pred_d or pred_e fails alone ⇒ recorded as partial structure, no adoption. **Nothing is adopted from
this rung**: §2912's configuration is untouched by it.

## Price

**1 full frontier pipeline run + 8 arms × 3 windows of forward evaluation, plus 32 one-off SVDs of 1152×1152 matrices, ≤ 600
GPU-seconds** (§2917's 7-arm run took 126.6 s; the SVDs are cached on CPU after first use and cost ~15 s once), 0 backwards, **0 fitted
parameters** — the arms rescale singular values of maps already fitted. The parent `ops/frontier_fisher8.py` is **unmodified**. Not
forward-instrumented, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 1`. Receipt:
`frontier_tail_spectral_split_results.json`, read with `price` in the same command the ledger section is written from, in the canonical
`Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
