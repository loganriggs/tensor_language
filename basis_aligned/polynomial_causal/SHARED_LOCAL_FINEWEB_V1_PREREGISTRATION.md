# Frozen output-factor validation on the existing FineWeb cache

13 September 2026. Read-only diagnostic, no fitting or model-body forwards. The G64 fit is frozen and missed its convergence/gain bars; G128 is still running. This protocol applies unchanged to each completed artifact, using its matched-storage global baseline (rank78 or167).

Reuse `/dev/shm/bilin18_frozen_radial_fineweb_v1.pt`, SHA256 `b1107a91d55df69bc05eefade10ffb362a5c1a81c783e7dfa3c746ea2dec72de`, positions63 and127 in64 historical FineWeb sequences. The next tokens are the targets. These rows were already opened in prior work: no fresh/OOD claim.

Let h be the cached native final residual, rho its RMS with native FP32 epsilon, b the cached final MLP output minus its known bias, U the native unembedding and Uhat the frozen approximation. Compare:

1. **Quadratic route only:** native raw logits plus `(Uhat-U)b/rho`. All other U readers remain native. This is the local coefficient-fit target; retaining U elsewhere prevents a whole-model U-storage saving.
2. **Whole unembedding:** `Uhat h/rho`. This could save the unembedding storage if fidelity holds, but applies the fitted map beyond its training objective.

Apply native `30*tanh(raw/30)` separately to every token in every arm. The same frozen final norm is correct because these interventions change only output readers, not residual states. Use FP64 arithmetic for comparisons and native FP32 replay as the instrument check. The cache's bias-subtracted branch includes its original rounding; independently replay the bias-free bilinear formula to measure that difference.

For each capacity, compare grouped and optimal-global programs under each route. The global optimum is computed only from the original folded weight metric, never from FineWeb.

- `pred_a`: native FP64/FP32 logit relative error <=1e-5, mean CE replay <=1e-4, and bilinear-branch unembedding-weighted replay <=1e-5.
- `pred_b`: grouped mean KL is no greater than matched-global mean KL on both routes.
- `pred_c`: grouped mean absolute per-position CE change and mean KL are each <=0.05 on both routes.

Keep misses, signed mean CE added (lower is better), mean absolute changes, worst changes, per-position CE/KL and both routes. A small signed average can conceal opposing errors. Neither an improved average nor a passed historical panel identifies selective circuits or completes native adoption. Artifact digests are recorded on execution; no factor selection or refit follows from this panel.
