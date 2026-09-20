# Baselines for simpler bilinear programs

A candidate must beat an executable baseline for the **same output and input ports**, at a stated error tolerance. Comparing a scalar component to the entire model is invalid. Compression alone does not establish a circuit.

## Required comparison ladder

| Baseline | Purpose | Current evidence |
| --- | --- | --- |
| Original native factors, exact normalization/attention/bias | Fidelity reference and original program price | v628 native replay |
| Frozen extracted dense recent path | Same h16/x0/v1 ports and same scalar target; independent execution | v628: 24,235,715 tensor values; parent has 256 signed squares |
| Zero / calibration-only constant / isotropic quadratic | Detect trivial prediction and normalization shortcuts | v623 rejects constant/isotropic explanations; rerun when target changes |
| Native bilinear-channel pruning, optionally scalar refit | Does simply retaining original units suffice? | v617/v618; global tensor scope only |
| Joint scalar quadratic eigendecomposition | Strong conventional baseline for a fixed output reader | v619–v624; 256-term parent, not yet a simple semantic circuit |
| Joint symmetric Tucker / HOSVD | Shared input/output spaces of the contracted third-order tensor | v615 family-specific bounds; not a universal circuit bound |
| HT on fixed trees, then alternative trees | Recursive quartic baseline with explicit slots and ranks | Toy controls exist; native HT comparison still missing |
| Sparse shared bilinear DAG | Candidate method: reuse, adaptive widths, sparse interactions | Reference quotient evaluator exists; automatic native discovery incomplete |
| Individual-matrix SVD, balanced-gauge SVD, matched-price random controls | Sanity/control families; not substitutes for joint decomposition | v629 raw SVD fails; native balanced-gauge and matched-price random arms not yet run |

Do not silently omit a missing baseline or describe planned runs as measured.

## Literal accounting

For one native MLP with d=1152 and h=4608: three dense matrices contain
3dh=15,925,248 scalar weights, plus 1152 output biases. It computes h=4608
bilinear channel products per token, in addition to linear projection arithmetic.
These are local costs; attention, residual sources and readout remain chargeable.

For a scalar reader c, form
Q = sym(L^T diag(D^T c) R), with scalar bias c^T b.
The exact eigendecomposition gives x^T Q x = sum_j lambda_j (p_j^T x)^2,
at most d signed squares. Dense eigendecomposition costs O(d^3) after forming Q;
forming Q directly costs O(h d^2). A truncated rank-r scalar program stores dr+r
values plus bias, but the reader/writer and upstream generators are charged if
needed for the declared interface. Shared factors are counted once in a DAG.
A rank-r eigentruncation is the coefficient-Frobenius baseline for this scalar
quadratic, not automatically the best behavioral predictor on normalized data.

Report storage (values, bytes, sparse indices), scalar products/squares, linear
arithmetic, distinct shared features, live state, and required external ports.
Use matched total price, not equal rank labels. Keep output dimension, precision,
context length, nonlinear operations and intervention semantics identical.

## Evidence and red-team rules

For every candidate report coefficient/polynomial error, scalar prediction error,
intervention-change error, behavioral effect, preservation controls and reuse
separately. Keep fresh, opened and replay panels distinct. Never average these
into a score that hides a failed requirement. Dense fallback is baseline success,
not compression success. Relative errors also need absolute numerator and target
norm when denominators may be small. Optimized methods need convergence evidence,
restarts and planted recoverability before a failure becomes scientific evidence.

Red-team successes with shuffled/matched-cost controls, held-out selection,
constant predictors, residual/background accounting and extraction without hidden
model access. Red-team failures with full-capacity replay, independent contraction,
bias/norm/axis/sign checks, precision, planted recoverable examples and factor
gauges. A row rescaling L_k -> a_k L_k, R_k -> R_k/a_k preserves the joint tensor,
but raw matrix SVD need not respect it. The executable gauge control shows why a
negative factor-SVD result cannot establish absence of a simple polynomial.

See [gauge control](DECOMPOSITION_GAUGE_CONTROL_RESULT.json),
[HT objective](HIERARCHICAL_TUCKER_SHARED_DAG_DIRECTION_2026-09-20.md),
[recent-path extraction](RECENT_FOLDED_COMPONENT_2026-09-20.md), and
[negative SVD comparison](RECENT_SHARED_FEATURES_2026-09-20.md).

## Native gauge/numerical follow-up

v630/v631 add the joint MLP16 input-mode Gram and balanced-factor controls.
The native CUDA SVD full basis fails orthogonality/replay; CPU float64 repairs
full-rank recovery but leaves rank512 failure unchanged. Joint-mode rank512
also fails. See [audit](JOINT_INPUT_BASELINE_AUDIT_2026-09-20.md). This is a
measured input-mode projection baseline, not yet a native HT or sparse-core run.

## Shared native-product follow-up

v632/v633 compare full-write, contracted-parent and matched random supports,
preserving full input span. The parent isotropic functional metric (including
trace) improves strongly over coefficient Frobenius selection but no compressed
arm reaches10% error. See [metric audit](SHARED_PRODUCT_METRIC_AUDIT_2026-09-20.md).
These are fixed-dictionary baselines, not a bound on learned quadratic features.

## Radial-feature extension

v634 adds one shared radial quadratic and centered native atoms; all compressed
arms fail. v635 fits calibration-only product means with the same supports and
improves error, but the best tested compressed validation error remains26%.
[Audit](RADIAL_FEATURE_AUDIT_2026-09-20.md) distinguishes isotropic weight algebra
from native-distribution statistics. Both remain negative candidates.

## Head-conditioned joint sparse-core baseline

v639/v640 natively verify the exact128-coordinate conditional normalized map
and compare joint tensor-energy frames with sparse versus dense cores. Both
output ranks32/128 fail native cross prediction even with dense cores.
[Measured frontier](HEAD_SHARED_CORE_BASELINE_2026-09-20.md) includes pair sharing,
branch prices and uncharged background dependencies; no complete-circuit claim.
