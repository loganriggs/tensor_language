# Weight-derived subspace discovery improves the wide quartic controls

2026-09-22 04:43 UTC. Five planted structures, two starts each. This is a positive control for the proposed decomposition pipeline, not a native circuit result.

**Recovering a relevant input subspace from the teacher weights gives 10/10 fits below 5% error; direct ambient fitting on the same rotated targets gives 5/10.** Median value errors are 1.02% and 4.90%, respectively; worst errors are 4.34% and 12.36%.

The targets are the same five structural families used in the preceding width test, but now their four relevant coordinates are hidden by a dense orthonormal embedding into 1,152 dimensions. The student is not given that embedding. The pipeline computes the teacher's exact Gaussian derivative Gram from its weights:

$$
H=\mathbb E_{x\sim\mathcal N(\mu,I)}[J_F(x)^\top J_F(x)].
$$

For a direction $v$,

$$
v^\top Hv=\mathbb E\|J_F(x)v\|^2.
$$

For a polynomial under a nondegenerate Gaussian, an exact zero means the directional derivative polynomial vanishes everywhere. Thus the nullspace consists of directions on which the polynomial does not depend; its orthogonal complement supplies an input subspace. This is a statement about the function, not a guarantee of a unique human-interpretable basis.

We select eigenvectors above $10^{-10}$ of the largest eigenvalue. All five cases yield rank four, with a large spectral gap. The recovered subspaces agree with the planted span within $1.5\times10^{-14}$ relative error. Direct polynomial replay after projection is below $7.1\times10^{-15}$. The student then fits the four-dimensional polynomial with the existing Adam at 0.1 for 250 steps procedure and profiled readout.

The primary prediction of at least eight recoveries passes: all ten succeed. The subsequent ambient control fits the **same densely rotated targets** directly in 1,152 dimensions, with the same Adam rate and budget. It gets five successes. Initial student functions are not identical across these different parameter spaces; this is a pipeline comparison, not an isolated causal ablation of one optimizer step.

The earlier axis-aligned wide test had only two Adam successes. Do not use that as the matched baseline here. Its difference from the dense-rotation control is itself a warning about coordinate-dependent optimization. The matched control was added after the reduced result and is explicitly descriptive, not a retroactive preregistered prediction.

## What is actually smaller?

The fitted toy program can retain a shared 1,152-by-4 projection,16 learned linear forms in that four-dimensional space, and a dense 2-by-4 output readout. This is 4,680 stored floating coefficients and 12 variable products. The direct four-atom wide representation uses 18,440 floats and the same 12 products. These counts include the projection; neither its coefficients nor its inference work are free. They exclude unrelated transformer operations because this is a two-output toy, not a deployed model. No runtime speedup has been benchmarked.

## Limits for the native work

The toy intentionally has exact intrinsic dimension four, and its teacher is available as four explicit quartic atoms. The native folded teacher is a much larger implicit two-layer computation. Computing its exact derivative Gram may be substantially more costly. Earlier text-residual sensitivity measurements were distributed across many input directions, and full-tensor mode spectra were broad. Those measurements are not the same derivative Gram or measure, but they prevent assuming the same rank-four reduction will hold.

This result therefore supports using weight-derived subspaces as a candidate-discovery step when spectra justify them. It does not repair the pending native fits, establish a small native subspace, identify semantic features, or validate OOD prediction/removal/composition. A truncated native subspace would need its own approximation and native-response tests. Existing native jobs remain unchanged.

The reduced experiment took 12.75 seconds; the matched ambient control took 14.21 seconds on two CPU threads. Full spectra near the cutoff, per-family errors and replay checks are retained.

[Protocol](WIDE_QUARTIC_SUBSPACE_PLAN_V1.md) · [Reduced results](WIDE_QUARTIC_SUBSPACE_V1.json) · [Matched-control protocol](WIDE_QUARTIC_SUBSPACE_AMBIENT_PLAN_V1.md) · [Ambient results](WIDE_QUARTIC_SUBSPACE_AMBIENT_V1.json) · [Implementation](toy_wide_quartic_subspace.py).
