# Full folded-path native replacement — 2026-09-21 01:00 UTC

Evaluate rank4 programs at 64 and 256 output widths, alongside exact projections onto those output subspaces and the calibration-mean predictor. The reference is the entire previous-MLP-source-dependent final bilinear contribution, not only the selected output features. Preserve its calibration mean. Replace its varying residual write in the actual final residual, then apply native final RMS normalization, unembedding, and softcap.

Measure CE added by replacement and centered-logit effect error for full removals and same-token swaps. Require teacher-coordinate replay <1e-8; register 256-output effect error <0.3 on both families/domains and replacement CE increase <0.05 nats on both domains. These are screens, not a definition of circuit identification. Exact projection comparators distinguish omitted output directions from fitted product error. Panels are reused diagnostics; no held data fitting. Forty-eight capture forwards, no selective 256-feature claim.

The upstream source and midpoint states remain supplied by the native model; the result is not an upstream MLP replacement or demonstrated end-to-end speedup.
