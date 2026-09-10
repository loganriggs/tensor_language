# Weight-based decomposition with two circuits and explicit normalization

The original direction is in
[bilinear_circuit_reconstruction_codex_handoff.md](bilinear_circuit_reconstruction_codex_handoff.md).
The [pilot report](bilinear_reconstruction_pilot_report.md) records the first
implementation and its negative discovery result. They are different documents:
the handoff proposes the approach; the report evaluates the bounded pilot.

**The weight-based route remains viable. The pilot ruled out particular local
span and duplicate-factor proposals, not decomposing a bilinear module after
folding in its actual writers and readers.** The next work should use both
circuits to constrain that decomposition and retain normalization explicitly.
The recent is/was command-mode sequence is paused at the user's request.

## What the existing work actually establishes

The repository already implements complete MLP-to-reader tensor contractions,
restricted input/output tensors, attention writer/reader folds, and exact
context-cross/self terms for an additive writer. These are in
[`subspace_weight_atlas.py`](../../bilinear_quotient/ops/subspace_weight_atlas.py).
We should extend that machinery instead of building another independent stack.

There are two relevant precedents:

- The v17 H3/MLP11 study stores an exact restricted `8×8×8` MLP tensor. Its
  original local causal gates passed; 32 selected native factors recovered
  about 81% of the held-panel MLP effect, with 18.9% tensor residual. The v18
  transfer passed the selected-factor and control gates but failed broad
  sufficiency of the shared input interface. Its terminal is
  `activation_specific_interface`. A restricted tensor is not a complete
  model of the module's use on new constructions.
- Temporal and is/was circuits use overlapping native modules but distinct
  response directions. Existing two-mode MLP weight comparisons found maximum
  quadratic overlap below .176 across the inspected sites. Separate causal
  task-mode studies also favored own-task over cross-task replacements. This
  supports task-specific functions inside shared locations. It does not rule
  out shared linear producers or a shared part of a larger product library.

Sources: [v17 tensor receipt](../../bilinear_quotient/circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json),
[v18 transfer](../../bilinear_quotient/circuits/followups/temporal_iswas_v18_frozen_shared_tensor_transfer_v1_result.json),
[two-task dossier](CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md).

## 1. Fold the question into the bilinear weights

For actual bilin18, an ungated bilinear MLP has

    f(x) = D[(L n(x)) ⊙ (R n(x))] + b,
    n(x) = x / sqrt(xᵀx/d + ε).

Here `d=1152`, `L,R` have shape `4608×1152`, `D` has shape `1152×4608`,
and `b` is the native output bias. The input to this expression is the state
immediately before the MLP's RMS normalization. The symbol `⊙` means multiply
corresponding entries. Actual deployed FP32 uses its native RMS epsilon;
changing the analysis dtype must not silently change that epsilon.

Let `C_A` and `C_B` be linear readers of the module output associated with two
circuits. Stack them into `C`. They may be validated output-coordinate maps
or actual downstream linear projections. A behavioral gradient is only a
local reader approximation and must be labeled accordingly.

For reader output `k`, define the symmetric matrix

    Q_k = sym[Lᵀ diag((C D)[k,:]) R],
    sym(M) = (M + Mᵀ)/2.

Then the exact local reader function is

    (C f(x))_k = xᵀ Q_k x / (xᵀx/d + ε) + (C b)_k.

This folds the output projection and reader into the multiplication weights.
The antisymmetric part contributes zero because both arguments are the same
`x`. We can contract only the needed readers or apply their matrices implicitly;
there is no reason to materialize the full `1152³` tensor.

This output-conditioned symmetric-matrix view, including eigendecomposition,
is developed in [Pearce et al., Sections 2–3](https://arxiv.org/html/2410.08417v2).
Their results motivate candidate input operations; they do not guarantee that
our checkpoint has a compact causally sufficient decomposition.

A spectral expansion gives

    xᵀQ_k x = sum_j λ_kj (v_kjᵀx)².

It identifies signed quadratic detectors used by a reader. Independently
choosing an eigenbasis for each reader can duplicate work or obscure shared
products. The two circuits should therefore be analyzed jointly.

## 2. Two different kinds of sharing must be distinguished

Consider two functions `g_A=x0*x1` and `g_B=x0*x2`. Their symmetric coefficient
matrices have zero Frobenius inner product, yet both reuse the linear
calculation `x0`. An angle between complete quadratic functions would miss it.
Conversely, two outputs can read the same quadratic function `x3²` with
different coefficients or opposite signs.

There are therefore at least three meaningful questions:

1. **Shared output functions:** do the spans of the two families of quadratic
   forms intersect? An intersection identifies a polynomial observable both
   readers can use, with their adapters kept explicit.
2. **Shared input calculations:** do the forms reuse linear factors or input
   subspaces even when their output-function spans do not intersect?
3. **Separate or interacting operations:** after identifying candidate inputs,
   are the cross-products between circuit-specific parts needed by either
   reader? Nonzero necessary cross-products argue for a joint operation.

The first question is a joint row-space problem on symmetric coefficient
matrices. The second is a factor-sharing problem; a low tensor angle cannot
answer it. The third is tested by the actual cross-contractions and by
independent/joint interventions. These decisions are more informative than
calling a whole head or MLP one circuit because both tasks use it.

CP decomposition seeks a shared product dictionary; Tucker decomposition
separates input/output coordinate maps from a core that may still contain
many interactions. Neither automatically identifies a circuit. Uniqueness
needs assumptions on the factors; a change of basis can otherwise produce
many descriptions of the same computation. The native hidden width4608 also
exceeds input width1152, so simple undercomplete simultaneous-diagonalization
arguments cannot be applied to all native factors without checking their
hypotheses. See [Kolda and Bader](https://www.kolda.net/publication/TensorReview.pdf).

## 3. A useful exact extension: retain the norm as a separate statistic

Here is a module-level construction derived for our normalized setting.
Let `U` have orthonormal columns and contain the column spaces of every `Q_k`
needed by both circuits. Then

    Q_k = U H_k Uᵀ,     H_k = Uᵀ Q_k U.

The reader outputs need only

    c = Uᵀx,     ρ = xᵀx,
    output_k = cᵀ H_k c / (ρ/d + ε) + (C b)_k.

The residual directions outside `U` still affect normalization. Keeping the
single scalar `ρ` preserves that dependence without reconstructing those
directions. This is an explicit nonlinear statistic, not a frozen normalizer.

Proof: substitute `Q_k=U H_k Uᵀ` into the numerator and retain the original
squared norm in the denominator. The identity holds for every real `x`, not
just fitted examples, whenever the stated matrix identity is exact.

For the unnormalized quadratic numerators, the smallest linear feature space
that suffices for all inputs is the span of all `range(Q_k)`. If a direction
`v` is discarded, invariance under `x→x+t v` for every `x,t` requires
`Q_k v=0` for every reader. This proves the necessity as well as sufficiency
of that linear space. It is not a minimum-complexity theorem for arbitrary
nonlinear encoders or the whole transformer.

The potential gain is conditional. If those ranges span all1152 dimensions,
there is no exact reduction of this form. A restricted eight-dimensional
input tensor cannot be used to pretend this full-domain condition passed.

For an approximate numerator, there is a simple weight-based local bound:

    |xᵀ(Q_k − Q̂_k)x| / (xᵀx/d + ε) ≤ d ||Q_k − Q̂_k||₂.

The matrix norm on the right is the largest absolute eigenvalue for the
symmetric error. This follows from the quadratic-form bound and the positive
denominator. It is a uniform local output bound, not a full-model KL or
intervention guarantee. Later nonlinear computation may amplify the error.

## 4. Independent interventions add concrete constraints

Suppose the two circuits supply additive writes `W_A a` and `W_B b` at this
boundary. These are explicit maps from their input coordinates; their native
producers and coefficients remain part of the program's cost.

Let `W=[W_A W_B]` and `z=[a;b]`. To update the norm after `x→x+Wz`, we need
`xᵀWz`. A sufficient intervention-closed feature space is

    range(U) ⊇ span(range(Q_1),...,range(Q_m),range(W_A),range(W_B)).

With `V=UᵀW`, the reduced state updates exactly as

    c' = c + Vz,
    ρ' = ρ + 2 cᵀVz + zᵀ WᵀW z.

The term `2 aᵀ W_Aᵀ W_B b` inside the final quadratic predicts overlap
between the writes. Sequential edits give the same reduced state as the
corresponding combined additive edit. The reduced reader then recomputes
its denominator from `ρ'`. An equivalent implementation can retain
`ρ_perp=||x−Uc||²` instead of total energy. Because legal edits lie in `U`,
this perpendicular energy stays fixed and the new norm is
`||c+Vz||²+ρ_perp`. This form avoids cancellation in the expanded energy
update; the current small reference tests the expanded identity.

This explains precisely why two circuits give additional information:
observing one reader permits more directions to be discarded; adding the
other reader and its independent edit maps removes that ambiguity. Shared
natural activity alone does not establish the required edit closure.

There are limits. The caller must supply the actual write being removed or
swapped; an undifferentiated total does not identify its source history.
Changing the writer mechanism itself can change `W`, which requires a new
closure check. This is a quotient of this module and this additive edit
family, not yet a state transition for the complete network.

## 5. Where attention can provide further folds

An attention writer has the form `O_h z_h`. If it enters the bilinear input,
its projection can be folded into each factor:

    L(x + O_h z) = Lx + (L O_h)z,
    R(x + O_h z) = Rx + (R O_h)z.

For two writers, the direct numerator cross-interaction is

    C D[(L O_A a)⊙(R O_B b) + (L O_B b)⊙(R O_A a)].

These are weights describing how two attention outputs jointly construct a
reader's feature inside the shared MLP. Context-cross, self, and cross-writer
terms all remain; keeping only the self term can miss most of a circuit's role.
If normalization intervenes, the corresponding norm scalar and its update
must remain explicit as above. The folded matrices do not justify moving a
linear map through an unchanged RMS operation.

In the other direction, an attention reader's Q/K/V projection can be folded
into `D` on an MLP-to-reader path. If residual additions or normalization lie
between them, include those operations and backgrounds explicitly. Q/K head
normalization and position rotations remain part of the reader.

For product attention, each matching score has a weight form such as
`x_tᵀ W_Qᵀ R_(t,s) W_K x_s`, with normalization factors retained separately.
The complete score is a product of two such matchers and a value read. A
shared intermediate can therefore be a query projection, key projection,
matching function, or value transformation—not necessarily an entire head.
Shared first-layer values alone do not make their routing functions equal.

The decomposition should preserve this factored graph. Expanding its global
fourth/fifth-order coefficient arrays would create avoidable size and conceal
existing shared operations.

## Executed mathematical control

[`normalized_bilinear_observable_reference.py`](../normalized_bilinear_observable_reference.py)
implements the contraction, reduced reader, norm statistic, and independent
writer updates. A64-dimensional planted module is rotated by an orthogonal
matrix, with weights/writers transformed consistently. Its two reader families
compute `(x0*x1,x3²)` and `(x0*x2,2*x3²)` before the common normalization.
Two writes include directions that affect only normalization as well as
numerator directions.

Six linear coordinates plus one norm scalar reproduce full reader outputs
and native, single, and joint edits to maximum absolute error **4.44e-15**.
All eight controls pass. Two negative controls are decisive: freezing the
norm gives wrong outputs; omitting the extra writer projections loses the
ability to predict edits even when the original reduced state is identical.
[Receipt](../NORMALIZED_BILINEAR_OBSERVABLE_CONTROLS_V1.json).

This is a planted instrument control, not a trained-model discovery or a
claim to beat its obvious sparse symbolic implementation.

## Concrete research decision

Prioritize the **joint writer–bilinear–reader tensor with explicit norm
closure**, using the two existing task circuits as constraints. The first
native object is MLP1, the lowest-index shared MLP in the established two-task
response interface. Reproduce the existing fixed four-coordinate task reader
maps from their recorded recipe; do not fit a new rank to obtain a favorable
answer. Analyze the joint reader functions and their upstream attention-write
maps together. Inspect shared polynomials, shared linear producers, and
cross-writer terms separately.

First establish exact contraction and whether this full input-domain observable
space is smaller. If not, report that result before considering an explicitly
restricted or approximate candidate. Any approximation must state its input
and edit domain and then predict both circuits' new-input outputs, separate
removals/swaps, and joint use. Charge dictionaries, input adapters, explicit
norm statistics, surviving native background and all arbitrary weights.

This is the next mathematical direction. The command-matcher continuation is
parked unrun. The earlier pilot's full-rank bank and the recent is/was nulls
remain evidence, but neither settles this two-circuit, normalization-aware
factorization question.

## Saved trained-weight check completed in this review

I also evaluated the actual saved v17 MLP11 factors, without loading a model
or running new task examples. The full restricted tensor contains an
antisymmetric component whose Frobenius norm is **40.4% of the original
tensor norm**. That component contributes exactly zero to `xᵀT_k x` over
real arithmetic. This is redundancy in the expanded tensor, not removal of
40.4% of native weights, computation, or behaviorally relevant energy.

On the symmetric functional tensor, the already selected32-factor residual
is **16.4%**, compared with the original unsymmetrized18.9% measurement.
The selection and old verdict remain unchanged. It is a correction of which
coefficient object measures the quadratic function, not a new trained circuit
or evidence that16.4% meets a stricter fidelity criterion.

Both saved attention writer maps, `L7H7` and `L9H4`, were folded into that
same tensor. The resulting `8×128×128` cross-writer tensor reproduces the
four-arm bilinear numerator interaction to **2.00e-11** on random coordinates;
its mixed term is live. Direct factor versus symmetric-function evaluation
agrees to **2.38e-13**. These are exact-form numerical checks within the saved
U8 interface. The writers have not been relabeled as the two semantic tasks,
and no new native normalized intervention has been tested.

[Audit code](../audit_saved_bilinear_function_tensor_v1.py) and
[result](../SAVED_BILINEAR_FUNCTION_TENSOR_V1_RESULT.json).
The bounded next native study is specified in the
[joint-reader protocol](../BILIN18_MLP1_JOINT_READER_WEIGHT_V1_PREREGISTRATION.md).

## A compact edit rule can exist even when the whole input space is needed

A further derivation separates two questions that the pilot could otherwise
conflate: can we simplify the computation from arbitrary inputs, and can we
simplify its response to a specified family of circuit edits? The second can
have a positive answer even when the first linear quotient fails.

Suppose the allowed input changes are `x -> x + Wz`. The columns of `W`
are the explicitly supplied write directions, and `z` gives their amplitudes.
For each symmetric reader matrix `Q_k`, retain:

\[
p_k=x^TQ_kx,\qquad g_k=W^TQ_kx,\qquad
h=W^Tx,\qquad \rho=x^Tx.
\]

Here `p` stores baseline quadratic numerators; `g` describes how each reader
responds to each write direction; `h` and `rho` retain the information needed
to update normalization. Precompute the small weight contractions
`H_k = W^T Q_k W` and `G = W^T W`. One edit updates these quantities exactly:

\[
\begin{aligned}
p'_k&=p_k+2g_k^Tz+z^TH_kz,\\
g'_k&=g_k+H_kz,\\
h'&=h+Gz,\\
\rho'&=\rho+2h^Tz+z^TGz.
\end{aligned}
\]

The output is still `p'_k / (rho'/d + epsilon) + bias_k`.
These identities follow by expanding each quadratic after substituting
`x + Wz`. Updating `g` and `h`, rather than freezing them after the first
edit, makes successive edits agree with their joint application.

For `m` readers and `r` fixed write directions, this state has
`m + mr + r + 1` scalars, regardless of residual width. This is a sufficient
state for that edit family, not a minimum-state theorem. A dense `Q_k` can
have full input rank and still permit this edit engine. If the write matrix
changes with the input or earlier interventions, that additional dependency
must be supplied and recomputed; the fixed-`W` theorem does not absorb it.

The link to the user's shared-module question is explicit. Partition `W`
into writes from circuits A and B. The off-diagonal block
`W_A^T Q_k W_B` determines their mixed numerator contribution, while
`W_A^T W_B` determines the mixed norm contribution. Even zero mixed blocks
do not by themselves imply additive normalized outputs, because the common
denominator also changes. These weight objects say exactly what joint
intervention needs to predict; module co-use alone cannot say this.

A new CPU control used eight random full-rank quadratic readers in96 input
dimensions and two write directions. All seven checks passed, including
independent-versus-joint edits, inverse edits, and deliberately omitted mixed
terms. The27-scalar edit state reproduced full outputs with maximum absolute
error **1.71e-13**. This is a numerical control of the derivation, not evidence
of a trained semantic circuit.

**The price caveat is essential:** initializing `p` and `g` still requires the
original input and quadratic functions. Their opaque weights and the native
prefix remain charged. A compact response engine could make causal testing
cheaper, but it does not constitute independent extraction or structural
model reduction. Discovery still needs an explicit reusable implementation
of those initial quantities, with the four requested behavioral properties.

[Executable edit reference](../bilinear_joint_edit_observable_reference.py)
and [seven-control receipt](../BILINEAR_JOINT_EDIT_OBSERVABLE_CONTROLS_V1.json).
The full-input MLP1 weight audit is now implemented and queued through the
managed GPU runner; no native result is claimed in this paragraph.

## The full-input trained MLP1 audit has now finished

The managed RTX5090 run took72.5seconds and exactly eight native forwards
(300 sequence evaluations). It restored the existing pooled eight-coordinate
output projector and each task's first four modes, and saved those readers
so subsequent coefficient work needs no new fit or task execution.

All registered numerical checks passed. The largest FP64 algebra discrepancy
was1.32e-11. Against captured native FP32 module outputs, the maximum absolute
reader-coordinate difference was4.48e-4, with relative discrepancy below
3.90e-7. The latter comparison is numerical agreement, not bitwise equality.

The eight reader functions jointly have numerical input support1152/1152.
Their smallest-to-largest singular ratio is0.03115, far above the registered
1e-10 numerical cutoff. The principal cosines between the two four-dimensional
quadratic-function spaces are0.4251,0.1944,0.0914,0.0506. A value of1 would
indicate a common function direction; none appears here. This uses the
coefficient Frobenius inner product, not a measured distribution of language
inputs or causal effects. The older report's0.176 maximum used two modes per
task, so these are different reader definitions, not contradictory reruns.

Both tasks have nonzero coefficients on all4,608 native hidden products.
That establishes native implementation co-use, but neither selective semantic
units nor an impossibility of a different shared factorization. The earlier
rank-four program's noisy behavioral miss also remains binding; these weight
checks do not promote its robustness status.

[Native result](../BILIN18_MLP1_JOINT_READER_WEIGHT_V1_RESULT.json) and
[serialized task readers](../BILIN18_MLP1_JOINT_READER_WEIGHT_V1_READERS.json).

There is an important scope correction before drawing a stronger conclusion.
Full input support rules out expressing all numerators using fewer *linear
coordinates alone*. It does not rule out using fewer linear coordinates **plus
the norm scalar**. For example, `Q=I` has full rank, but its numerator is just
`rho`. Treating the previous result's terminal name as ruling out every
linear-plus-norm representation would overstate what that test measured.

The correct condition for a representation from `(U^T x,rho)` is

\[
Q_k=U H_k U^T+\alpha_k I
\]

for every reader, where `U` has orthonormal columns. To see necessity, rotate
or reflect the component of `x` perpendicular to `U`. The proposed state
cannot distinguish these inputs. Thus the quadratic can contain no cross
term between retained and discarded coordinates, and its action on the
entire discarded space must be a scalar multiple of the identity. These
conditions also suffice, since the numerator becomes
`(U^T x)^T H_k (U^T x) + alpha_k*rho`.

Consequently, any discarded direction must be a common eigenvector of every
reader matrix. For two matrices, define their **commutator** as
`K=Q_A Q_B-Q_B Q_A`: their actions in the two possible orders, subtracted.
Every common eigenvector lies in the kernel of `K`. If `K` is invertible,
there is no discarded direction and no proper linear-plus-norm quotient,
even with arbitrary nonlinear decoding of the retained state.

The next managed run fixes the first temporal and first is/was reader and
tests that obstruction exactly. Binary floating-point coefficients are
rational numbers with power-of-two denominators. They can be mapped into
arithmetic modulo the prime65,521. A nonzero determinant there proves the
original rational determinant is nonzero; a zero result is inconclusive.
Small Fraction-arithmetic controls, a determinant oracle, and the norm-only
counterexample all pass. The native certificate has been queued with zero
model forwards and an independent integer determinant replay. This tests one
representation class; it is not a claim that the model lacks smaller nonlinear
circuits or domain-specific semantic computations.

## Exact result and what the two readers now let us construct

The exact test has finished. Its determinant is **12,024 modulo65,521**,
which is nonzero. Independent GPU and CPU elimination agree, and all controls
pass. Thus these fixed MLP1 reader functions admit **no proper linear-coordinate
plus norm representation over all real inputs**. This closes that representation
class, including the norm exception above. It does not rule out a smaller
nonlinear arithmetic program, an appropriate task-domain representation, or
an exact edit-response engine with native initialization retained.

[Certificate result](../BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1_RESULT.json).
The run used no model forwards and took2.86seconds including independent
integer verification. It treats the saved binary coefficients as exact rational
numbers; it is not a theorem about bitwise equality of every native FP32 trace.

A different use of two circuits sharing the module is now concrete. Stack their
output readers as `C=[C_A;C_B]`. Their **dual write map** is

\[
D=C^T(CC^T)^{-1},\qquad CD=I.
\]

Its columns specify the smallest output changes that set the requested reader
coordinates. Changing the module output by `D_A z` changes A's coordinates by
`z` and leaves B's coordinates unchanged. The corresponding component maps
`P_A=D_A C_A` and `P_B=D_B C_B` are generally *oblique projectors*: they select
components along complementary directions rather than perpendicular directions.
They satisfy `P_A P_B=P_B P_A=0`; local removals therefore commute.

This is information the second circuit supplies: preserving its reads adds
constraints to the first circuit's write. Ordinary separate orthogonal patches
can alter the other reader; their largest cross-reader gain here is0.3234.
The constrained alternative requires at most **5.68% more write magnitude**
than an unconstrained edit, across every possible four-coordinate edit amplitude.
All saved-reader algebra checks pass, maximum edit-replay error8.22e-15.

This construction does not guarantee that the final behavior of the other task
is preserved. Later layers can read different coordinates and react nonlinearly.
It also does not discover new semantic features or reduce the model's opaque
weights. Its immediate value is a precise, inexpensive candidate intervention
for deciding whether the shared native module can be separated along the two
existing task computations.

The [native selectivity protocol](../BILIN18_MLP1_DUAL_READER_NATIVE_V1_PREREGISTRATION.md)
fixes48 previously opened target pairs and16 temporal P-control pairs, compares
ordinary and constrained edits, and recomputes the full suffix. No new fit is
allowed. The intervention primitive passes seven controls; the native test is
not yet run. The [saved-reader result](../BILIN18_MLP1_DUAL_READER_EDIT_V1_RESULT.json)
is a local tool-feasibility result, not circuit identification or adoption.

## Native outcome: useful task separation, failed control comparison

The18-forward native test has now run. Its numerical and intervention checks
pass, including identical full-vocabulary outputs for the two equivalent joint
implementations. On the48 target pairs, the constrained edits retain over98%
of the ordinary own-task effects. Their signed projections onto the native
margin-change vectors are0.5200 for temporal and0.5747 for is/was, versus
0.5229 and0.5831 for the ordinary edits. These are effect projections, not
prediction accuracies or fractions of all model behavior explained.

Cross-task margin-change RMS is0.454% and2.760% of the respective intended
own-task RMS. Thus the local constraints coexist with a useful task split in
the native suffix. However, the **registered control predicate fails**: on the
16 temporal P controls, the temporal edit's mean teacher KL rises from
0.000157899 to0.000167102 nats/token, exceeding the permitted1e-6 increase.
All P top-token predictions stay unchanged. The small magnitude does not
turn the failed no-worse criterion into a pass, and iswas-specific controls
remain untested. No circuit is promoted.

[Native behavioral receipt](../BILIN18_MLP1_DUAL_READER_NATIVE_V1_RESULT.json).
Execution took2.43seconds after data preparation, with no fitting or updates.

The next question moves from coordinates to operations. Let `u` be the native
pre-attention state at layer1, and `a` its new attention write. Neither stream
is assumed to be purely token or context information. Each task reader sees
three explicit numerator operations: `u` multiplied with itself, `a` multiplied
with itself, and the symmetric `u`-times-`a` cross term. All three retain the
same full normalization denominator. The next hypothesis is that the cross
operation supplies both task-typed effects while meeting the original control
bar; the other pieces are measured without being fallback candidates.

This connects directly to earlier source-pair work, including an earlier
failed attempt to promote attention1's contribution as a portable MLP1 path.
The proposed test edits a node of an explicitly expanded output computation.
It does not claim that removing that node equals removing attention at its
physical input, or that separately normalized paths add. Seven new algebra
and capture controls pass; the native cross-program test is still pending.
See the [fixed protocol](../BILIN18_MLP1_ATTENTION_CROSS_PROGRAM_V1_PREREGISTRATION.md).
