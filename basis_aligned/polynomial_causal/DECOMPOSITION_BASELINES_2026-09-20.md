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

## Data-informed output-objective control

v641 compares the same dense-core output ranks with a calibration-interaction
eigenspace and random frame. The empirical frame has2%calibration error at128
but18–68%cross error on other families. [Audit](HEAD_OUTPUT_OBJECTIVE_AUDIT_2026-09-20.md).
This fitted-basis failure is not a proof against all native-distribution metrics.

## Task-specific effect baseline

The subject-number nine-edge graph costs92 native block evaluations per sequence;
the exact same five-port joint effect costs36 using two suffix corners. An
aggregate-only algebraic query plan reduces92 to85 but remains more expensive.
Individual edge attribution is a different interface and needs its own comparison.
[Cost and reuse audit](SUBJECT_NUMBER_COST_AND_REUSE_AUDIT_2026-09-20.md) verifies
the symbolic cancellation and distinguishes same-task transfer from cross-task reuse.

v643/v644 now support a conditional six-MLP response baseline: freezing six later
attention writes gives1.7–3.9% effect error on opened prompts and3.0–9.8% on new
matched-position prompts. [Conditional chain](SUBJECT_NUMBER_CONDITIONAL_CHAIN_2026-09-20.md)
charges eight vector ports,95,560,716 fixed values and27,648 products. It is an
explicit dense reference for future folding, not a simple discovered circuit.

v648 validates the513,030-value width8 joint response compiler after correcting
baseline-port precision, but state-SVD8 fails the effect gates. v649 identifies
large losses even at initial-only and final-only projections. v650 reserves two
answer readers at the same width/cost and improves strongly, but still fails.
[Observable-basis audit](SUBJECT_OBSERVABLE_BASIS_AUDIT_2026-09-20.md) preserves
these negatives and separates oracle diagnostics from executable predictors.

v651/v652 use derivative/finite-reader bases; v653 uses separate snapshot-balanced
encoders and decoders. All retain width8 and fail the all-cell effect gates. The
dual-space program costs522,246 values, including its extra initial encoder.
[Finite-reader audit](FINITE_READERS_AND_DUAL_SPACES_2026-09-20.md) distinguishes
exact contraction, planted recovery and empirical native prediction.

## Conditional positive and its limits

v654 broadens calibration coverage at unchanged width8 and passes the conditional
5% effect gate in every opened evaluation cell (1.2–2.9%). Full-native fidelity
still fails two cells (10.2–11.3%). v655 exports and independently replays the
shared-product runtime on CPU; native context generation remains charged.
[Portable program](SHARED_SUBJECT_RESPONSE_PROGRAM_2026-09-20.md).

v657 restores attention12 while keeping all suffix MLPs dense. Native restoration
recovers only17.9–24.3% of the postposed frozen-model error, missing the50% gate.
The projected restoration passes10% full-native effect error on these opened rows.
This is not yet composition with the reduced MLP chain.
[Positive and negative audit](ATTENTION_RESTORATION_AUDIT_2026-09-20.md).

v658/v660 compose the attention12 fold and reduced MLP chain. Both pass the5% conditional gate; full-native10% still fails one cell on each panel. v660 adds genuinely new longer structures under frozen bases. v659 corrects the producer export by including55,296 output-encoder values. [Composition, transfer and accounting](COMPOSED_SUBJECT_RESPONSE_TRANSFER_2026-09-20.md).

## Same-width preservation and sparse interaction controls

v665–v667 use four calibration reader contrasts at width8 and pass prospective target/control tests. v668 compares18/36 and9/36 shared-pair supports, three equal-price random18 supports and a zero-quadratic-numerator baseline. Half-support passes; quarter/zero fail; one random also passes. Dense normalization and native context costs remain. [Definition, geometry, prices and verdicts](JOINT_READER_SPARSE_RESPONSE_2026-09-20.md).

## Conditional source-amplitude baseline (later20 September)

[Native source Hessians](SEMANTIC_SOURCE_QUADRATIC_JET_2026-09-20.md) provide a measured same-output/source-interface comparison: linear3, diagonal6, signed-rank1 7, full symmetric9 values/context. Rank1 andfull pass unit/negative/mixed10% on opened contexts; doubled edits fail. Producer computation and context dependence are fully charged; this is not a global tensor-decomposition or HT comparison. CPU extraction is exact only for the quadratic observable.

Later [quadratic design audit](SHARED_SOURCE_QUADRATIC_DICTIONARY_2026-09-20.md): original four arms had symmetric designrank2/6. On six independent source settings, fullquadratic4.72% passes, rank1 15.74% fails. Always report monomial-design rank; signed/strengthened versions of the same direction do not identify additional Hessian dimensions. Sharedoutput2/input2 fail17.33/18.29% on this stronger benchmark.

## Full-path curvature controls

[Exact local Hessian sum](NATIVE_SOURCE_CURVATURE_DECOMPOSITION_2026-09-20.md) supplies a native coefficient reference for curvature-path omission. Full quadratic passes fifteen positive single/pair directions; linear, all-MLP-only, attention-only and zero-A/B-cross controls fail their stated gates. Joint omission of individually passing terms fails modal fidelity. These are conditional derivative-program baselines, not native HT or whole-model replacements.


20 September: [Exact source attention fold and spectral baseline](ATTENTION_FOLD_AND_SPECTRAL_BASELINE_2026-09-20.md). Native attention fold replays exactly but fails the one-shot speed gate (1.024x). Per-context signed rank-two Hessians pass opened finite-edit tests with68 versus80 stored values; native generators remain required. Rank-one fails. No native HT or complete-circuit claim.

20 September: [Shared five-source features](SHARED_FIVE_SOURCE_FEATURES_2026-09-20.md) fail the reuse screen: common rank2 plane17.46% heldout number error; per-output planes do not rescue it; three calibrated sparse residual pairs improve to11.86% but still fail. Full-basis recovery is exact. Opened data only.
