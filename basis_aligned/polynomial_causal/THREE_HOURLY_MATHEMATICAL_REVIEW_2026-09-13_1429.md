# Mathematical review — 13 September14:29 UTC

## Actual object and what the last cycle changed

The retained parent uses two1152-to128 Q/K maps, a shared64-dimensional source
projector, one current-value reader and a fixed1152-dimensional writer. Query
and source slots repeat in the numerator: degree two in each, followed by a
source value factor. Original RMS denominators, rounded positions, causal
summation and nonlinear suffix remain explicit, so the full program is not a
single polynomial tensor. Conditional input/background generators remain priced
dependencies rather than explained away.

The last cycle's complete-error insight led to a useful sparse reader:64nodes,
55,296 retained reader entries, full-rank Gram correction. Weight-only fitting
on the composed numerator passes corpus aggregates and manipulation tests, while
uniform basis fitting does not. Local stationarity, structural storage, native
preservation and runtime performance are separate conclusions. CSR execution
fails despite packed savings; regular two-of-four fits also fail fidelity.

## Literature mapping and prior-result check

Matrix interpolative decomposition represents a matrix using selected columns
and an interpolation matrix containing an identity block. Apply that construction
to $B^T\in\mathbb R^{64\times1152}$, where $B$ is the current orthonormal source
basis. This selects64source-coordinate rows of $B$. We use deterministic pivoted
QR, not the randomized algorithm or its probabilistic error bounds. Full column
rank permits an exact chart; measured conditioning controls the numerical concern.
The pivots are not unique or semantic identifiers. See
[Martinsson, Rokhlin and Tygert](https://users.oden.utexas.edu/~pgm/Pubs/2011_ACHA_random1.pdf).

The tensor counterpart selects terms from an existing canonical separated
representation, using matrix interpolation machinery. It is relevant if a large
explicit product dictionary is already the target. Our rational normalized
parent is not directly such a tensor, and expanding its compact products can
increase cost. We do not infer a cheaper parent from that algorithm alone. See
[Biagioni and the Beylkins](https://arxiv.org/abs/1306.5013).

This is not a newly discovered coordinate trick: the project's
`INTERACTION_COORDINATE_COMPLEMENT_V1_MATH.md` already uses pivoted coordinate
charts for a retained subspace in another interaction. The executable question
here is whether adapting it to the small64-dimensional parent improves its
shared QK execution without any approximation. Generic CP/Tucker compression,
TT hierarchy and Hankel realization remain distinct routes reviewed previously;
none supplies an established smaller normalized parent under our current ports.

## Exact contraction and executable consequence

Let $I$ be64pivot rows, $J$ the remaining1088rows, and $E=B_I$ invertible.
Set $C=BE^{-1}$, so $C_I=I_{64}$. For source batch $X\in\mathbb R^{n\times1152}$,

$$
Z=XC=X_I+X_JC_J,\qquad XB=ZE.
$$

For either key map $K_i\in\mathbb R^{128\times1152}$,

$$
XBB^TK_i^T=Z(K_iBE^T)^T.
$$

Both QK factors reuse the same $Z$. Their precompiled key adapters stay64-by128;
no runtime inverse is needed. The projector is unchanged, so all value,
normalization and position operations may remain exactly where they were.
Only the linear source contraction changes. There is no new nonlinear factor,
rank reduction or assumed input distribution. The approximation norm is numerical
replay of both inside-key outputs, on independent and cached native inputs.

Pivot selection and construction cost $O(dk^2+k^3)$ for$d=1152,k=64$; each source
projection replaces$ndk$ multiply-adds with$n(d-k)k$ plus gathers/additions.
Reader storage drops by$k^2=4096$ coefficients, but adapters and coordinate indices
must also be charged. A full-rank chart is exact algebraically; this is not a
minimal arithmetic-circuit theorem. Ordinary pivoted QR does not inherit every
strong rank-revealing bound, so condition and finite-precision replay are measured.

## Executed result and decision

`PARENT_COORDINATE_CHART_V1_RESULT.json` records pivot condition11.64 and maximum
interpolation coefficient1.085. Replay errors are below3.18e-15 in FP64 and
5.83e-7 in FP32. Prepared FP32 state, including both key adapters and int64
pivot/complement indices, falls from360,448 to353,280bytes, about1.99%.

Two-thread CPU speed ratios are0.662–0.701 in FP64 and0.419–0.528 in FP32 across
1/19/128/512source rows. The1.1-times speed prediction fails at every size.
The exact coordinate chart therefore gives a small storage/edge simplification,
but does not beat shared dense execution. It is not adopted as a speed improvement.
No native semantic rerun is needed to claim the local identity; no unmeasured
full-model or GPU speed claim is made.

The consequence has been executed, not merely proposed. Retain the validated
packed component; stop escalating local runtime engineering without a larger
shared-dependency or contraction change. Before transferring sparse fitting to
another interaction, use its dossier and explicit graph to identify what can
actually disappear and what all consumers still need. The full four-property
model goal remains incomplete. Next mathematical review due17:29UTC.
