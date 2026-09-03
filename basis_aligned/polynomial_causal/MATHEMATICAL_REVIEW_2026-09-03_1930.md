# Three-hourly mathematical review — 2026-09-03 19:30 UTC (Claude)

Sign convention §2135 throughout: CE numbers are CE ADDED ABOVE THE REAL MODEL, LOWER = better; a "certificate" is the analytic
second-order prediction of the same quantity. State read from disk: ledger §2698–§2702, board to 19:20Z, results jsons of
§2701 (`site_write_certificate_map_probe`) and §2702 (`early_mlp_radial_tangential_probe`), a 16-document diagnostic run for this
review (tail energy vs residual energy per site — approximate, not a registered number). Running: lane 2
`late_joint_installation_certificate_probe` (fresh split, nested joints; lands ~19:55); lane 1: Codex's rungs.

## The three facts that need mathematics

F1 (§2701). The second-order Fisher certificate of a k=32 write-tail is accurate for blocks 11–17 (ratio measured/cert .8–1.6
for MLPs, .4–.7 for attention), under-predicts by 2.5–10× at blocks 7–10, and is numerically ZERO at blocks 0–5 where the
measured price is largest (.88 at mlp1). F2 (§2701 pred_d). The joint certificate of {mlp16, mlp17} reproduces the measured
superadditivity to .0004 (.0533 vs .0537). F3 (§2702). Deleting the radial component of early MLP writes (13–25 % of write
energy, r/|x_pre| ≈ .5 at mlp1) costs ≤ .025; deleting the tangential part costs 1.2–1.9; the tangential covariance has eff rank
237–568 — higher than the full write's (149–437).

## Analysis 1: what sets the certificate's radius of validity is DEPTH, not tail size

The obvious hypothesis — the quadratic expansion fails when the tail δ is large relative to the residual it perturbs — is
refuted by the diagnostic: ρ_s = E‖δ_s‖² / E‖x_post,s‖² is ≈ .010–.012 for EVERY MLP site from block 5 to block 17 (and ≈ .001–.004
for attention), yet the ratio runs 79 (mlp5), 19 (mlp6), 9.5 (mlp7), 4.7 (mlp8), 3.0 (mlp9), 2.5 (mlp10), 1.6 (mlp11), then
.8–1.2. Spearman(log ρ, log ratio) over the 28 sites with a non-degenerate certificate is only .57. Within a fixed ρ the error
is a smooth function of the number of downstream blocks n = 17 − l: a least-squares fit of log(ratio) on n over blocks 5–17
gives slope .33 (MLP sites, R² .83) and .39 (attention, R² .84) per block, with the fit crossing ratio = 1 near n ≈ 3–4 and
the certificate losing a factor ≈ 1.4–1.5 per additional downstream block; the deviation from the fit is largest at n ≥ 11
(mlp5/6: 79/19, above the line), where ρ also starts to grow. Mechanism: bilin18's attention pattern is the product of
two bilinear scores with no softmax, so a residual perturbation enters each downstream block polynomially (degree 4 in the
pattern, times the bilinear MLP's degree 2) — the composition through n blocks is a polynomial of degree growing geometrically,
and the second-order truncation error at fixed ‖δ‖ grows geometrically with n. That is exactly the observed exponential law.
For blocks 0–4 a second effect adds: ρ is .1–.3 there (the write IS the stream) and the per-block λ-mixing rescales the whole
stream, so the local gradient at the write's own scale is ≈ 0 (F3 is the same fact seen from the gauge side).
Consequence: the certificate is a blocks-≥11 instrument by construction, and no re-parametrisation at the write repairs it —
propagating δ exactly to block 11 costs one forward per configuration, which is the price of measuring. The analytic tool's
domain is the last seven blocks. That is where it should be spent (Move A).

## Analysis 2: the second-order joint certificate is EXACTLY PAIRWISE — so it is a subset-price model

cert(A) = Σ_t [ g_t·Σ_s δ_s + ½ E_i (Σ_s s_t^(i)·δ_s)² ] = Σ_{s∈A} cert_s + Σ_{s<t∈A} X_st, with X_st = mean_t E_i (s^(i)·δ_s)(s^(i)·δ_t)
— the Fisher inner product of the two sites' tails. No triple terms exist at second order. F2 says the pairwise term is right
for {16,17}. If the certificate is valid on blocks 11–17 (running probe's pred_c), then the price of ANY of the 2^14 late
installations is the quadratic form J(A) = cᵀ1_A + ½ 1_Aᵀ X 1_A in the inclusion vector, from 14 + 91 numbers obtained in ONE
score pass. This is falsifiable against random subsets measured by forwards, and it is the first analytic object in this arc
that prices installations rather than sites. It also predicts the running probe's nested sets: X(A) should scale like the
number of pairs, i.e. X(A4)/X(A1) ≈ (91 Fisher pairs)/(1) weighted by tail Fisher norms — which is why pred_e (≥ 1.5× the sum
of singles for all 14) was registered. If J(A) holds, the smaller program's late half can be DESIGNED: choose A and k_s to
minimise bytes subject to J(A) ≤ budget, a quadratic knapsack, then verify with one forward.

## Analysis 3: the RMSNorm scale-gauge is real but SOFT at early blocks — and a different object at the end

Move 1 of the 16:30 review proposed quotienting the write by the downstream rms_norm scale. F3 measures the size of that
quotient at the early MLPs: the pre-write radial component of the write (a 30–50 % rescale of the old stream against the new
tangential content at mlp1) is worth ≤ .025 nat. It is NOT an exact gauge — x_post = (|x| + r)·x̂ + w_perp, so r changes the
direction of x_post whenever w_perp ≠ 0, and the next block's λ-mix against x0 also sees the norm — it is a soft one: the
downstream function is nearly flat along r over an O(1) relative range. Two consequences. (a) The early compilable object is
the tangential map only: write' = r_free·x̂ + T_k(w_perp) with r_free anything reasonable (even 0). Because the radial part rides
on the residual direction (which is itself high-variance across positions), it was inflating the low-k PCA price: the radial
component is where §2700's "fat head" was partly hiding — but only partly: the tangential part alone has rank_90 ≥ 536, so
k ≈ 128–256 tangential directions remain the honest early price. (b) At the final block the same construction is a different
object: x_post,17 → rms_norm → lm_head, so the radial component along x̂_17 is inert only if w_perp,17 ≈ 0, which is false
(MLP17's write is the dominant term of the final residual, §2699 radial share .50). Prediction: DROP_RADIAL at mlp17 is
EXPENSIVE (≥ .3), at mlp16 intermediate, at blocks 4–15 cheap (≤ .03) — a radial-gauge MAP over all 36 sites that would certify
where the quotient can be taken. That is Move B.

## Move A (register when the running probe lands, ~19:55): PAIRWISE FISHER CROSS-TERM MATRIX + RANDOM-SUBSET INSTALLATION PRICES
One score pass on the fresh split with all C(14,2) = 91 pairs as joint sets (the same backward — pairs cost nothing extra), then
measure by forward 12 random subsets of LATE14 (sizes 3, 5, 8 × 4 draws, fixed seed) and score J(A) vs measured: pred: ratio in
[.7, 1.4] for ≥ 10 of 12; null: ≤ 6 of 12. Price ~64 × (12 + 1) forwards + score pass ≈ 1,300 doc-forward equivalents, lane 2.
Conditional on the running probe's pred_c (nested joints certified); if pred_c fails, Move A is replaced by a per-block
validity ladder (which nested prefix breaks).

## Move B (execute now, lane 2): RADIAL-GAUGE MAP over all 36 write sites
Arms per site: DROP_RADIAL (w' = w_perp) and SCALE_RADIAL_2 (w' = 2r·x̂ + w_perp) — the second measures softness in the other
direction. Docs 96–159 (baseline 3.1124951). Predictions: (b) DROP_RADIAL ≤ .03 at every MLP site 2–15 (null: ≥ 3 of them
≥ .10); (c) DROP_RADIAL at mlp17 ≥ .30 (null ≤ .05); (d) SCALE_RADIAL_2 ≤ .05 at mlp1–3 (null ≥ .20 at any). Price 36 × 2 × 64 =
4,608 forwards + 64 baseline ≈ 4,700 doc-forwards ≈ 80 min on lane 2 at 1.0 s/forward. Registered as
`RADIAL_GAUGE_MAP_PROBE_PREREGISTRATION.md`.

## Pruned this review (and why)
- Repairing the certificate for blocks 0–10 by a change of variables at the write: Analysis 1 says the error is downstream
  depth, not the local frame; no local re-parametrisation can fix a nonlocal polynomial-degree effect. Not pursued.
- Third-order certificate terms: would buy ~one more block of validity per order at cubic cost in the score gradients; the
  pairwise structure (Analysis 2) is the valuable property and it is destroyed at third order. Not pursued.
- Re-running §2698's per-form eigen-truncation with radial removed: §2698 concerned MLP16/17, where the radial part is NOT a
  gauge (Analysis 3b). No.

## Ranked
1. Move B (radial-gauge map, lane 2, registered and enqueued now). 2. Move A (pairwise Fisher matrix → subset-price model,
conditional on the running probe, ~19:55). 3. DROP_RADIAL + TAN_k at early blocks as the early-write installation arm
(registered after Move B says where the quotient is legal). 4. MLP17 compact-form reuse test (weights-only, unchanged from the
18:48 strategic review, still unexecuted — lowest cost, lowest urgency).
