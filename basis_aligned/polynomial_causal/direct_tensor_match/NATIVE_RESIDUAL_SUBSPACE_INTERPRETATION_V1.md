# Native residual: the compact Gaussian-gradient subspace prediction fails

2026-09-22 04:46 UTC. New native-weight CPU computation; no candidate fit or new text capture.

**A 64-dimensional input subspace captures only 43–45% of the missing small-output computation’s derivative energy on independent Gaussian probes.** The registered 90% prediction fails in both parent dictionaries. The rank-four positive toy therefore does not justify assuming a similarly compact subspace for the native residual.

The object is the exact pure quartic MLP16→MLP17 path minus each frozen mixed-CP512 approximation, keeping output coordinates 4–15. These are the twelve smaller fixed output coordinates, not semantic labels or native MLP channels. We use the same calibration-derived mean, covariance and capped inverse-output-energy weighting as the pending residual learner. Write

$$
x=\mu+Sz,\qquad R(z)=F(\mu+Sz)-\widehat F(\mu+Sz),
$$

and estimate the weighted input derivative Gram

$$
H=\mathbb E[J_R(z)^\top WJ_R(z)].
$$

Here $z$ is standard Gaussian, $W$ is diagonal on the twelve selected outputs, and $S$ is the covariance square root. Derivatives are in whitened coordinates. We do not project out radial directions: this is the learner’s Gaussian law, not the earlier text-state tangent metric.

We estimate $H$ from 128 synthetic Gaussian probes and evaluate the resulting eigenvector subspaces on 128 independent probes, with seeds fixed before computation. Native Jacobians and parent Jacobians use the existing analytic routines. No text output labels select a basis.

| Input rank | Training capture, seed 1001 | Checking capture, seed 1001 | Training capture, seed 1002 | Checking capture, seed 1002 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 15.3% | 14.9% | 16.2% | 15.6% |
| 16 | 28.9% | 24.8% | 30.4% | 26.2% |
| 32 | 40.4% | 32.7% | 42.0% | 34.3% |
| 64 | 55.5% | 43.2% | 57.0% | 44.8% |
| 128 | 72.3% | 56.3% | 73.5% | 57.8% |
| 256 | 87.6% | 71.2% | 88.1% | 72.3% |
| 512 | 97.2% | 86.2% | 97.3% | 86.8% |

Capture means projected squared derivative magnitude divided by total squared derivative magnitude, with the same output weights. It is not function-value reconstruction accuracy, a fraction of recovered semantic features, or a native intervention score. The result file includes every output separately.

The train/check gap matters: rank 512 captures roughly 97% on the fitting probes but only 86–87% on independent probes. This is a Monte Carlo subspace estimate, unlike the exact derivative Gram in the planted controls. It may overfit its probe set. We have falsified the stated learned-subspace prediction, not certified the globally best 64-dimensional population subspace or proved a lower bound on every compact circuit.

Native quartic Euler-identity errors are below 1e-10, and the independent finite-difference check is below 1e-6; exact values are in the receipt. The sample derivative Gram is positive semidefinite by construction up to floating-point error. Input moments and output weights retain their existing calibration-data provenance.

**Decision:** do not introduce a rank-4 or rank-64 native input restriction on the strength of the successful toys. Preserve the already queued unrestricted-direction residual learner and evaluate its actual finite responses. If a future subspace approach is proposed, it must justify its dimension with independent native evidence and charge its projection. These measurements still leave room for sparse nonlinear computation distributed across many linear input directions.

[Protocol](NATIVE_RESIDUAL_SUBSPACE_PLAN_V1.md) · [All spectra and per-output captures](NATIVE_RESIDUAL_SUBSPACE_V1.json) · [Audit](audit_native_residual_subspace.py) · [Contrasting positive toy control](WIDE_QUARTIC_SUBSPACE_INTERPRETATION_V1.md).

## Larger-probe follow-up: the broad residual persists

2026-09-22 04:49 UTC. Repeated the same measure and output subset with 1,024 fitting and 1,024 independent checking probes, new fixed seeds. This addresses the substantial subspace overfitting in the first screen; it does not replace the original record.

| Rank | Training/checking capture, parent 1001 | Training/checking capture, parent 1002 |
| ---: | ---: | ---: |
| 4 | 15.4% / 15.5% | 16.1% / 16.2% |
| 16 | 27.4% / 27.1% | 28.6% / 28.4% |
| 32 | 37.2% / 36.4% | 38.5% / 37.7% |
| 64 | 49.9% / 48.2% | 51.3% / 49.7% |
| 128 | 64.9% / 62.3% | 66.1% / 63.6% |
| 256 | 80.3% / 77.3% | 81.1% / 78.3% |
| 512 | 93.0% / 91.0% | 93.3% / 91.4% |

Rank 64 still captures only 48–50% on checking probes; its fitting/checking gap shrinks to roughly two percentage points. Rank 512 now captures 90.95/91.38% on checking probes. This supports the conclusion that this residual is distributed across many directions in the specified Gaussian derivative geometry, while explicitly improving the finite-probe estimate. It is still not a certified optimal population subspace.

Capturing 91% of derivative energy does not mean 91% of function values are reconstructed, nor a 9% relative reconstruction error. A function approximation would require its own construction and value/response tests. A broad linear span can also support a sparse nonlinear program, so this result does not rule out economical DAGs. No compact-subspace replacement is exported.

The follow-up took 25.47 seconds on two CPU threads. [Separate protocol](NATIVE_RESIDUAL_SUBSPACE_LARGE_PLAN_V1.md) · [All updated spectra and output captures](NATIVE_RESIDUAL_SUBSPACE_LARGE_V1.json).
