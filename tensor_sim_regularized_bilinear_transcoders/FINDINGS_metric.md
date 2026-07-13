# Metric verification — findings before the training loop

Ran the sanity suite (`sanity_checks.py`, all pass) on the closed-form Gaussian tensor inner product
(`tensor_sim.py`) before writing any transcoder training loop, per the handoff's tip. Two things worth
knowing, one of them a **bug in the handoff's recipe**.

## What the metric is (and the identity that makes it testable)

`⟨A|Λ|Â⟩ := E_{x̃}[y·ŷ]`. Via Wick/Isserlis this is closed-form in the CP factors (4 matmuls + 2 Hadamards,
`O(r r' d)`, no `K×d×d` tensor ever built). The consequence worth leaning on:

> **`L_fid = ‖A−Â‖²_Λ / ‖A‖²_Λ = E‖y−ŷ‖² / E‖y‖²`** — the *relative Gaussian-expected MSE*, in closed form.

That gives an independent ground truth to test against (Monte-Carlo), which is how both findings below surfaced.
Verified: closed form == exact brute-force Isserlis contraction to **1.4e-16**, and == MC to ~1e-3 (MC noise).

## FINDING 1 (bug) — the "data-matched Σ" recipe is wrong for lifted inputs

The handoff says: *"Estimate empirical covariance Σ of the lifted inputs once, then plug into the Gram
recursion (G⁰ = Σ instead of I)."* **This is silently, badly wrong.**

Isserlis requires a **zero-mean** Gaussian. The lifted `x̃ = (1, x)` is not one — its first coordinate is
deterministic. Plugging the *uncentered second moment* `E[x̃x̃ᵀ]` into the zero-mean Wick formula predicts

    E[x̃₀⁴] = 3·Σ₀₀² = 3      but the truth is  E[1⁴] = 1

Measured consequence on a random bilinear layer over lifted inputs: **37–95% relative error** in `⟨A|Λ|Â⟩`
(vs. Monte-Carlo truth). It would have silently biased every fidelity number.

**Fix (implemented, `tensor_inner_mean` / `fid_loss_mean`):** `x̃=(1,x)` *is* still a (degenerate) Gaussian —
mean `μ̃=(1,μ_x)`, **centered** covariance `Σ̃` with a zero row/col on the constant coord. So use the
**non-central Wick** 4th moment:

    E[xᵢxⱼxₐx_b] = ΣᵢⱼΣₐ_b + Σᵢₐ Σⱼ_b + Σᵢ_b Σⱼₐ
                 + μᵢμⱼΣₐ_b + μₐμ_bΣᵢⱼ + μᵢμₐΣⱼ_b + μᵢμ_bΣⱼₐ + μⱼμₐΣᵢ_b + μⱼμ_bΣᵢₐ
                 + μᵢμⱼμₐμ_b

It contracts to a single `r×r'` kernel — **same `O(r r' d)` cost**, still data-free per step, still
differentiable — and reduces to the centered formula when `μ=0`. Verified exact against MC (rel err 1e-4).
Use `lifted_moments(x)` to get `(Σ, μ)` correctly from raw activations.

## FINDING 2 (clarification) — the gauge group is permutation × rescaling, NOT `GL(r)`

The handoff says "insert `U`, `U⁻¹` on the hidden index." A **general invertible `U` is not a gauge of a CP
tensor** — only permutation and rescaling of the hidden index are (which the handoff's parenthetical does say).
Verified both ways, and this is a control that *must* fail:

| transformation | `L_fid` | status |
|---|---|---|
| hidden permutation | 0 (exact) | invariant ✓ |
| hidden rescaling (`L_h→aL_h, R_h→bR_h, D→D/(ab)`) | 0 (exact) | invariant ✓ |
| **L↔R swap** | 0 (exact) | invariant ✓ — *extra* gauge the handoff doesn't mention: it transposes each slice, and a quadratic form only sees the symmetric part, so the layer's **function is literally unchanged** (verified `forward(L,R)==forward(R,L)`) |
| random invertible `U` on hidden index | **27.9** | **breaks — as it must.** If a metric were invariant here it would be degenerate. |

## Status

`tensor_sim.py` (metric) + `sanity_checks.py` (9 check groups, all pass) are verified on: random tiny layers
(exact brute force), a jacclust DGP hand-built bilinear layer, and a **real 500M bilinear MLP layer**
(`r=4608, d=1152`; closed-form `‖A‖²_Λ` matches MC `E‖y‖²` to 7e-4). Safe to build the training loop on.

**Recommendation for E1 onward:** use `fid_loss_mean` with `lifted_moments(x)` from real activations (not the
centered `G=Σ` recipe), and keep the U-control in the test suite as a regression guard.
