# Single-head normalization causal screen

The preceding normalization screen passed, but still computes all attention heads.
New candidate g2 = block8 mixed residual + attention8.2 only; freeze MLP8 norm to
mean(g2²)+FP32 epsilon. Preserve both ordered crosses, write quadratic, residual
skip, donor-conditioned head8.2 delta, and full recursive native suffix.
Candidate implementation computes only head8.2; full package arrays remain loaded,
so current storage saving is zero. No fitting, scalar calibration or new source selection.

Dataset: frozen HEAD2_MLP8_CROSS_FRESH_V1_ROWS, now opened: 20 cells /40 sequences,
240 endpoint rows. CPU diagnostic has been inspected: local errors 2.1–6.4%; no
causal outcomes for this candidate inspected. This is a screen, not fresh confirmation.

Four arms: native, native8_midpoint, exact_reduced, single_head_norm =160 body
forwards, timeout300s. Reuse endpoint batching, native hooks, and original anchors.
pred_a: native/native8/exact_reduced replay saved fresh[0,1,3] <=1e-4 absolute AND
<=1e-5 relative L2, finite outputs, zero outside mask, exactly160 forwards.
pred_b: candidate target effect predicts exact reduced effect <=.35 relative L2
in EVERY family, with candidate target RMS>=1e-5. Failed family kills adoption.
pred_c: all four unrelated reader RMS changes <=.5 candidate target RMS in EVERY
family. Null: exact/full-context denominator reference; random-direction and fresh
selectivity comparison explicitly deferred, not inherited from old variant.

Measure effect error also against full native8 half-write descriptively. No change
to the certified exact package, no composition claim, no token-only claim. Passing
nominates fresh confirmation of this fixed formula against baselines and same-site
norm-matched nulls; failure retains full context and ends this omission route.
