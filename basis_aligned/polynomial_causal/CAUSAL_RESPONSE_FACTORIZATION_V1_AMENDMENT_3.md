# Causal-response factorization v1 — amendment 3

Status: controlling and frozen before any FIT response bundle is deserialized. This
amendment changes only the numerical device and dtype of the fixed Adam optimization
loop. It does not change candidate ranks, seeds, objective, prices, train/validation
roles, selection gates, or claims.

## Preserved inefficiency found prospectively

On a synthetic tensor with the exact production shape
`[2 phases, 49 sources, 49 targets, 229 training documents]`, a representative
shared/private candidate with global rank 8 and owner-private rank 2 took
18.916 seconds for five CPU float64 Adam steps: 3.783 seconds per step and 795,848 KiB
peak RSS. At the frozen 2,000 steps this extrapolates to about 7,566 seconds, or 2.10
hours, for one seed of one candidate. The full three-seed grid would therefore be an
avoidable multi-day computation.

This benchmark used random synthetic responses and opened no FIT or EVAL artifact.
The failed `/usr/bin/time` invocation preceding it performed no computation because
that executable was absent; the successful timing came from Python's monotonic clock
and `resource.getrusage`.

## Accelerated but mathematically identical objective

The optimizer still minimizes the full masked training MSE

$$
\mathcal L(\theta,H)=
\frac{1}{|\Omega_{\rm train}|}
\sum_{(p,s,t,d)\in\Omega_{\rm train}}
\left(\widehat R_{pstd}(\theta,H)-R_{pstd}\right)^2
$$

with Adam, 2,000 steps, learning rate 0.03, and the three frozen seeds. The only
change is that parameters, response, and mask inside Adam use CUDA float32. Random
initial parameters are generated first on CPU float64 and then cast, giving every
device the same seed-defined mathematical preimage.

All artifact validation, signed-response construction, canonical factors, document
codes, initial and final MSE, improvement fraction, replay, validation scores, and
Pareto comparisons remain CPU float64. After optimization, factors are copied to CPU
float64, exact scale/sign/permutation canonicalization is applied, and the reported
loss is recomputed from the canonical tensor program—not accepted from the float32
optimizer scalar.

## Numerical gates

- Every seed must remain finite and improve by at least the preregistered fraction.
- Canonical CPU float64 replay is the authoritative fitted loss.
- Seed spread is reported and a one-seed apparent win is unhealthy.
- The planted shared-plus-private toy must retain greater than 0.9999 improvement and
  final CPU float64 MSE below $10^{-8}$ under the float32 loop.
- Same-seed CPU float32 execution must replay exactly in the source-isolated test.
- Before production fitting, one synthetic production-shape CUDA timing and memory
  receipt must be preserved. A speedup that fails the planted or replay gates is
  rejected regardless of wall time.

This amendment does not authorize the analysis lifecycle. The amended optimizer and
its source closure require independent review before touching receipt-bound FIT
values.
