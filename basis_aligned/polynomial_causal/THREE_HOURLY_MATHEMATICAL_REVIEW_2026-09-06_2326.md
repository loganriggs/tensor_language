# Three-hour mathematical tensor-network review — 2026-09-06 23:26 UTC

## Decision

The current object is not “the lowest complement-loss DAS axis.” It is a finite causal
input/output operator with upstream interventions as rows and independently varied downstream
readers as columns. Noise or KL regularization may improve estimation, but cannot identify this
operator when the training loss observes only one task-specific column. The next DAS comparison
must therefore freeze each estimator on lexically and construction-disjoint training rows and
score a held-out *vector* of causal effects, projector stability, and complement inertness. In
parallel, continue literal weight compilation of the natural MLP8 -> attention9 -> Q8 circuit.

## Exact current objects

The checkpoint has residual width `d=1152`, `h=9` heads of width `d_h=128`, 18 blocks, and
vocabulary output `V=50,257`. For an MLP8 post-cue hidden intervention
`Delta h_8 in R^{B x T x m}`, its literal residual write is

\[
 \Delta x_8[b,t,:]=M[b,t]W^{Down}_8\Delta h_8[b,t,:]\in\mathbb R^{1152},
\]

where `M` is the registered post-cue mask. With complete attention and MLP responses at blocks
9 through `L-1` clamped to base, the pre-attention state at `L in {11,15}` is exactly

\[
 z_L'=z_L+g_L\Delta x_8,\qquad
 g_L=\prod_{k=9}^{L}\lambda_{k,0},\qquad c_L'=\operatorname{RMS}(z_L').
\]

The first full-clamp run verifies the normalized-state equation to `1.9e-6` at L11 and `3.8e-6`
at L15. Its v1 value equation was implemented incorrectly. The actual shared value bus is the
layer-0 value `v^(1)`, so

\[
 V_L=(1-\eta_L)W^V_Lc_L+\eta_Lv^{(1)},\quad
 \Delta V_L=(1-\eta_L)W^V_L(c_L'-c_L).
\]

V2 now tests that exact equation and contracts the result with base attention patterns over the
post-cue sources and the literal selected `W^O` head blocks. There is no fitted coefficient. The
full model is analytic/rational rather than polynomial because of RMS, attention normalization,
and tanh; the frozen-response residual propagation is degree one in `Delta x_8`, while the RMS
reader is nonlinear and the bilinear MLP writer is degree two before normalization.

For DAS, let `U in Gr(r,1152)` be a rank-`r` subspace, `a_i` an upstream donor/base command, and
`v_j` a downstream context/readout. Define the exact centered causal response tensor

\[
 H_{ijq}(U)=\ell_q(\operatorname{do}_{U}(a_i),v_j)-
             \ell_q(\operatorname{do}_{0}(a_i),v_j),\qquad q=1,\ldots,V,
\]

or its declared Q8/readout contraction. A scalar task loss observes only
`sum_q w_q H_ijq` on a small training block. Any rotation or added direction in the nullspace of
those observed contractions is invisible. This is the formal memorization channel: a lower
training complement loss need not mean a more correct causal subspace.

## Gauge, identification, and price

`U` is identified only by its projector `P_U`; bases transform as `Q -> QG`, `G in O(r)`. A
writer/reader factorization `H=CR^T` has the larger gauge `C -> CG`, `R -> R G^{-T}`. Stable
identification therefore means held-out projector or causal-operator agreement, not coordinate
agreement. Difference-in-means supplies a fixed estimator; constrained DAS searches many more
degrees of freedom and consequently needs stronger prospective controls.

Literal direct-skip storage is the MLP8 Down tensor plus existing block scalars, RMS, c_v,
patterns, and c_proj weights; the compiler adds no learned floats. Its current execution price is
seven forwards over 30 rows. A regularized DAS fit adds optimization steps and `1152r` basis
parameters, so it is justified only if it improves held-out causal operator prediction or stable
identification—not merely the training scalar.

## Theorem/algorithm mapping

For a rational series, a complete Hankel block factorization `H=PS` recovers a minimal weighted
automaton; minimal state dimension is Hankel rank, with factorizations unique up to similarity.
Spectral algorithms implement the recovery with an SVD and pseudoinverses. Rabusseau, Li, and
Precup additionally show equivalence between weighted automata and linear second-order RNNs and
recover linear 2-RNN tensors from low-rank Hankel tensors
([AISTATS 2019](https://proceedings.mlr.press/v89/rabusseau19a.html)). This maps exactly to our
*finite measured restriction*: upstream commands are prefixes/rows, downstream contexts and
readouts are suffixes/columns, and `H_ij` is the causal effect. It does not prove the transformer
is a rational series or that our finite command family is reachable and observable.

Regularization can improve finite-sample system identification bounds—Sun, Oymak, and Fazel
analyze this for low-order linear systems
([L4DC 2020](https://proceedings.mlr.press/v120/sun20a.html))—but their guarantee assumes a
specified linear dynamical object and controls estimation error. It does not turn a one-column
behavioral objective into an identifiable multicolumn causal operator. Thus noise/KL is a useful
estimator guardrail here, not a substitute for held-out reachability/observability.

The finite causal matrix already measured across temporal and is/was commands has one frozen Q8
realization predicting all four task quadrants at cosine at least `.9999966` and relative RMSE at
most `.262%`; its numerical rank is 4 while commands span 8 coordinates. This gives a finite
4--8 state bracket. It is stronger shared-state evidence than any within-task scalar DAS score.

## Executable consequence

Run a prospective estimator tournament with identical rank and train rows:

1. difference-in-means;
2. complement-loss DAS;
3. DAS with intervention noise;
4. DAS with KL-to-DIM or projector anchoring.

Freeze all four before evaluation. Score them on disjoint cue pairs, lexicon, construction, and
downstream readers using (a) full-vocabulary causal-vector cosine/RMSE, (b) Q8/shared-state effect,
(c) complement inertness on A/P/C controls, (d) projector stability across train splits, and (e)
joint composition. The regularization hypothesis predicts noise/KL improves these held-out
operator metrics and reduces seed variance even if training complement loss worsens. The
wrong-target hypothesis predicts optimized methods continue to win only the training scalar,
while DIM or the tensor-derived Q8 wins the held-out vector/operator test. Kill any estimator as
a circuit identifier if it cannot predict the sealed block without refitting.

This tournament is not the immediate GPU successor because V2 is already queued and can close a
literal mechanism at much lower ambiguity. After V2, either compile the material MLP9 correction
through its bilinear factors or launch the sealed estimator tournament. Do not fresh-confirm the
unpenalized six-MLP greedy union as a minimal circuit.

Next mathematical review due around **2026-09-07 02:26 UTC**.
