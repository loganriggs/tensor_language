# Exact conditional attention baseline

The branch-factored compiler retains Q, K, Q2, K2 as affine functions of the shared input coordinates before multiplying their scores. It avoids storing the expanded score coefficients indexed by both token positions and both latent coordinates. Input RMS, head RMS, rounded RoPE, causal masking, value mixing and cached first-layer values remain explicit.

The same conditional input/output boundary and all factors are charged. Storage has no T²r² coefficient array; execution still uses quadratic attention scores. This is an exact recursive bilinear computation baseline, not a newly identified semantic circuit or a source-reuse rescue. Background preparation remains external.

CPU evidence: six tests passed in 2.03 seconds across the new compiler, finite attention readers, full suffix readers and QR balancing. The compiler matches an independent native implementation and the expanded compiler at absolute/relative tolerance 1e-10 with nonorthogonal bases, non-negligible normalization epsilons, rounded RoPE and multiple signed amplitudes. Future-token perturbations preserve earlier outputs.

For B=2,T=32,d=12,H=3,r=3,a=2, the expanded program stores 145,543 tensor values and the factorized program 4,395. These counts exclude scalar metadata equally; they are a toy storage comparison, not a model-wide compression result or runtime benchmark. Native-model replay and export integration remain untested.

Implementation: `factorized_two_qk_attention.py`; checks: `test_factorized_two_qk_attention.py`. The supported v665 frame is unchanged. Failed v671–678 source-preservation results remain in `SOURCE_REUSE_AND_FULL_READERS_2026-09-20.md`.

## Native validation and cost reversal

v679 checks all attention layers12–17 on48opened rows: output relative error4.10e-7, finite-response relative error1.43e-5, expanded float64 difference8.88e-16, causal leakage0. All three registered gates pass. Inner execution1.48s (not an isolated compiler speed benchmark).

The factored program is larger on these5–8token batches: 1.39–2.13 times the expanded tensor count. Retain expanded storage for this short-context circuit. Exact shape formulas, independently checked against all24 compiled programs, predict crossover at length12 for batch12 and length7 for batch1, with d1152,H9,r8,a8. Longer-length counts are analytical extrapolations, not native runtime tests; see FACTORIZED_ATTENTION_STORAGE_2026-09-20.json. No source-reuse or whole-model result follows from this algebraic equivalence.
