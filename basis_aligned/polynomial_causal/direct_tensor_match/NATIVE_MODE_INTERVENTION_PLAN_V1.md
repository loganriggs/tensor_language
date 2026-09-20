# Canonical feature removal in the native model

2026-09-20 21:36 UTC. Motivation: whole-branch preservation passed the
ablation-relative bar but narrowly failed its absolute CE threshold. The next
question is whether individual extracted scalar computations predict removal
of corresponding components of the native branch.

Use the four fixed orthonormal directions U and fixed calibration output mean
mu from CANONICAL_ROOT_FEATURES_V1.pt. For reduced native branch y and student
prediction yhat, define native mode a_g=U_g^T(y-mu), student mode
ahat_g=U_g^T(yhat-mu). Independently remove scale*R_u^{-1}U_g*a_g/s(h)^2
or scale*R_u^{-1}U_g*ahat_g/s(h)^2 from the ORIGINAL native residual state.
Both use exactly the same native background; this isolates mode prediction
from errors in all other student modes. Keep the native final RMSNorm/softcap.

This is an operational feature definition, not a semantic concept. Projection
chooses the native comparison; it does not establish that these are intrinsic
unique units. No monosemanticity or behavioral selectivity claim is permitted.

Use new cached FineWeb documents80:96, context128. No refitting. All four modes
and their joint removal are mandatory; report every mode, including weak ones.
Compare full post-softcap logit-change vectors and token-level CE-change vectors.
Predictions registered before native mode outcomes:
- pred_a_replay: exact zero edit and joint sum of residual mode edits agree
  with independent projection within relative1e-5.
- pred_b_major: modes0/1 logit-effect cosine>.90 and relative effect error<.40.
- pred_c_all: all four modes logit-effect cosine>.80 and relative error<.65.
Also report signed CE, KL, argmax agreement and joint-removal effect error.
Do not confuse additive residual edits with additive nonlinear output effects.

Prior CPU scalar check on REUSED isolated panels: mode errors18–20%,26%,37%,
53–54%; correlations .98,.98,.95,.85. These motivated the bars but are not
independent evidence for native intervention. A failure can mean downstream
normalization changes geometry or that amplitude mismatch matters more than
correlation. Report effect magnitude so small/inert changes cannot pass by
absolute tolerance. Price includes four analysis-mode projections; no claim
that they are free or that subtraction instrumentation speeds up inference.
