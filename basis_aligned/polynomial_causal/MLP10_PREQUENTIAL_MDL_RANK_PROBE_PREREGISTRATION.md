# Parallel probe: prequential/MDL cross-validated rank of MLP10's effect matrix (noise-robust coverage credit)

**Status:** prospectively frozen after §2666 (top-3 captures 0.76 of reliable variance but with a soft 0.58
noise floor), before any description length is computed. CPU-only, zero forwards, zero deployed parameters.
Owner: Claude parallel lane. Math-review move #1 (0708). Replaces §2666's effect-variance fraction with a
rigorous, noise-robust prequential/MDL number, and red-teams whether §2666's coverage survives a proper
description-length test. Not a certificate/frontier claim (§2135 unused).

## The mathematics

§2666's coverage fraction used positive-eigenvalue variance, which a 32-dim noise matrix already inflates to
0.58. The rigorous fix is cross-validation / prequential coding (Dawid 1984; Rissanen MDL): fit the top-r
circuit subspace on document half0, then CODE half1's effects with it. Because the two halves have independent
noise, half1's energy inside half0's subspace is SIGNAL only — held-out captured energy has no noise-floor
inflation. The description length `DL(r) = (n/2) ln(RSS1(r)/n) + (r*32/2) ln n` (BIC form; `n=83*32=2656`,
`RSS1(r)=||M1(I-P_r)||_F^2`, `P_r` = top-r circuit subspace of half0) penalises overfitting automatically and
its minimiser is the MDL-optimal effective rank. Bits saved vs mean-only `= (DL(0)-DL(r*))/ln 2`.

## Object (frozen, from the rung520 discovery bundle)

Bundle `mlp10_source_star_causal_quotient_rung520_bundle.pt` (`7838deca…`). Per-node per-half 32-circuit effect
reconstructed exactly as §2657-§2666 (validated: reproduces material_nodes=83). `M0,M1 in R^{83x32}` over the 83
material nodes, circuit columns mean-centred over nodes. `P_r = V_r V_r^T` where `V_r` are the top-r right
singular vectors of `M0` (the half0 circuit subspace). Held-out residual `RSS1(r)=||M1 - M1 P_r||_F^2`;
symmetric `RSS0(r)=||M0 - M0 Q_r||_F^2` with `Q_r` from M1 (report the average to remove split asymmetry).
Cross-validated captured fraction `g(r) = 1 - RSS1(r)/||M1||_F^2`. `DL(r)` as above. r ranges 0..12.

## Frozen predictions (with measured bars)

- **A — instrument.** Bundle SHA256 `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06` and
  §2666 result SHA256 `9d2cdc37719e3554fc2b375c5f6e27b50d0fb24ca074a0315a35ddc4b4d5172f`; reproduces
  `material_nodes==83`; `||M1||_F^2 > 0`.

- **B — the MDL-optimal effective rank is LOW.** `r* = argmin_r DL(r) <= 6`, and `1 <= r* ` (a nonzero
  low-rank object). This is the overfitting-robust confirmation (or refutation) of §2658's rank-3 finding.

- **C — the low-rank SAVES bits and captures the majority of held-out (cross-validated) energy.** `DL(r*) <
  DL(0)` (net positive bits saved vs mean-only), AND the cross-validated captured fraction at r=3 is a MAJORITY:
  `g(3) >= 0.50`. (`g(3)` is the noise-robust replacement for §2666's 0.76; because it is held-out it cannot be
  inflated by a noise floor, so this is the honest coverage number.)

`strong_null = not (A and B and C)`.

## Reading and routes (frozen)

- A false: repair only the reconstruction clause.
- A true, B false: MLP10's effect matrix is NOT low-rank by prequential MDL (r* large or 0) — §2658/§2666's
  low-rank picture does not survive an overfitting-robust test; a major red-team result, report r* and the DL
  curve.
- A,B true, C false: low-rank but the held-out captured fraction is a minority — §2666's 0.76 was inflated by
  the noise floor; report the honest (smaller) g(3) and revise the coverage claim.
- A,B,C true: the MLP10 effect matrix is genuinely low-rank (r*<=6) with majority held-out coverage and positive
  bits saved. Report r*, g(3), and bits saved as the noise-robust coverage-credit number superseding §2666's
  soft fraction; hand the bits-saved figure (additive across results) to the coverage-credit accounting.

Assumptions that may fail: Gaussian residual coding approximates the signed-CE residual (prequential eval guards
mis-specification); BIC's `ln n` penalty is asymptotic (report the DL curve so the choice is visible); the 83
nodes are action-correlated (a subspace, not iid, but the held-out coding is agnostic to that).

## Literal price

Zero forwards, zero backwards, zero deployed parameters. Two SVDs + 13 projections; CPU, < 1 s.

## Frozen inputs

- rung520 bundle SHA256: `7838deca6432f76af14d3ef9f363c5d783bf70490fa199ce00a7b84aa3b19a06`
- §2666 result SHA256: `9d2cdc37719e3554fc2b375c5f6e27b50d0fb24ca074a0315a35ddc4b4d5172f`
