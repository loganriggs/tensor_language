# Mathematical restart review — 2026-09-09 13:48 UTC

## Exact object and prior null

The controlling handoff calls for a previously unspecified reusable operation with
explicit inputs and consumers, OOD/full-output extraction, removal and composition
evidence, and reduced structural description including opaque content. The completed
local-span null concerns linear update coordinates and proportional product factors.
It does not exclude a short nonlinear rule such as equality.

The next object is the first attention layer of the existing hop model: vocabulary29
(24 entities and5 fixed special tokens), D128, four heads of width32, context240.
It computes normalized token embedding n(a), four linear maps, native rotary maps
R(t), two dot products multiplied and divided by32^2, causal source sum, and native
value/output maps. For exact real-arithmetic rotations, at d=t-s>=0,

K[h,d,a,b] = ((R(d)Q1_h n(a)) dot K1_h n(b))
             ((R(d)Q2_h n(a)) dot K2_h n(b)) / 32^2.

The head output is sum_s K[h,t-s,a_t,a_s] V_h n(a_s), followed by W_O and residual
lerp. Downstream MLP and attention remain native and live. This is degree four in
exposed normalized query/source vectors, degree five including a value factor;
it is not a polynomial in tokens or the unnormalized residual. RMSNorm is explicit.
The parameter representation has ordinary reciprocal Q/K gauges and head-coordinate
changes compatible with RoPE; the scalar K is insensitive to gauges preserving those
products. The proposed symmetry acts on input entity names, not on hidden coordinates.

## Literature mapping and exact derivation

Deep Sets characterizes permutation-equivariant linear maps by tied diagonal and
off-diagonal coefficients (Lemma3). Our restricted entity-by-entity kernel has exactly
that form if it commutes with every entity permutation. The paper does not establish
that a trained transformer has the symmetry; this is the assumption being tested.
Source: [Zaheer et al., 2017](https://papers.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf).

Maron et al. characterize invariant/equivariant tensor bases using equality patterns
of indices. This supports using orbit indicators instead of coordinate SVD to specify
the candidate grammar. Our five fixed special labels add directed mixed and fixed-pair
orbits; we derive those below rather than borrow a dimension formula for a different
group action. Source: [Maron et al., 2019](https://arxiv.org/abs/1812.09902).

Let S24 act simultaneously on a and b, leaving the five specials fixed. Two ordered
entity pairs are in the same orbit precisely when they have the same equality status.
Transitivity maps any entity to any other, and any ordered unequal pair to any other
ordered unequal pair. Mixed pairs are specified by direction and the fixed special;
special-special pairs are each fixed. Thus there are exactly2+5+5+25=37 orbits.
The indicator functions of these disjoint orbits form a basis of invariant kernels.

For each head and lag, the uniform Frobenius-optimal invariant approximation is
Kbar[a,b]=mean{K[i,j] : orbit(i,j)=orbit(a,b)}. Proof: squared error is a sum of37
independent scalar least-squares problems; each unique minimizer is its orbit mean.
Equivalently it is the group average |S24|^-1 sum_g K[g(a),g(b)], computed by orbit
means without factorial enumeration. Uniqueness holds for positive uniform weights
on all ordered token pairs, not for an unseen task-weighted behavioral objective.
Exact representability is certified only if K=Kbar; arbitrary trained coefficients
need not meet it. Residual Frobenius size alone is not a causal or KL error bound.

Compute the weight-derived kernel in O(H T V^2 P) time and O(H T V^2) storage, then
aggregate the orbits in O(H T V^2). The current simple Boolean implementation loops
over37 fixed orbits. No full-model polynomial, global Hankel matrix, or dense hidden
tensor is constructed. The kernel has807360 FP64 scalars (6458880 bytes); the candidate
has35520 scalar coefficients instead of65536 first-layer Q/K weights. All remaining
335104 native parameters are separately charged. This is a proposed 30016-constant
structural saving with a shared explicit predicate, not a precision saving or a claim
that35520 unexplained positional constants have become understood.

## What the theorem does not solve

The learned model need not be name-invariant internally even when the target algorithm
is. A later layer could cancel asymmetric routing. Normalized value embeddings still
contain arbitrary learned content. First-layer K is token-local; later layers are
contextual, so applying this token table there would repeat the historical a1v error.
Native rotary tables contain FP32 rounding: dependence only on t-s is a real-arithmetic
identity, not bitwise deployed equivalence. The unreduced lag-table control separately
tests that error before any symmetry verdict. Discrete token interventions are supported;
arbitrary continuous changes to Q/K inputs are outside this extraction's contract.

CLUE/minimum linear interfaces would not identify this nonlinear equality predicate;
the full-rank result already motivates changing the grammar. Weighted-automata minimal
realization does not immediately apply to the contextual normalized suffix. Joint
polynomial tensor factorization remains an alternative but carries larger identification
and optimization ambiguities. The finite-group projection exactly solves this bounded
candidate-fitting problem without solving end-to-end mechanistic equivalence.

## Executable consequence and decision

`equality_router_reference.py` implements orbit construction, projection, positive,
perturbation, generic and independent-consumer controls, and a complete token-to-logit
program with the original first-layer Q/K weights physically removed. Four head
consumers reuse the predicate while keeping their coefficients and editable edges
distinct. `run_equality_router_v1.py` tests its full outputs and native-corresponding
single/joint equality-edge removals with the suffix recomputed.

Opposing predictions: a sufficiently invariant implementation preserves full
distributions and the removal/interaction vectors at the registered bars; necessary
entity-specific routes fail those bars, even if an attention plot or task accuracy
looks good. In either case the result says something about the proposed operation.
The prior08control CPU checks pass, including perturbation rejection9.52e-7 and
independent routers sharing one payload remaining different. No checkpoint outcome
has been inspected at the time of this review. The managed experiment is the next
step; a toy positive alone cannot license any model-level claim.
