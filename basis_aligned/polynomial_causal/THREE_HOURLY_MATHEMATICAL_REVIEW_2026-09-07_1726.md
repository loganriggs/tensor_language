# Three-hour mathematical tensor-network review — 2026-09-07 17:26 UTC

## Decision

The nested-DAS and v15 module results identify the same mathematical defect: the current observation
map is too small.  Six scalar DAS losses can improve while dropping an L15 causal response, and the
old response coordinates call a five-MLP native patch nearly complete (`.928` projection, `.029`
RSE) when behavioral recovery is only `.399`.  The next object is therefore a finite *causal-response
operator*, not another optimized axis or lower-rank activation basis.  Its rows are construction,
example, and downstream-test blocks; its columns are complete module/head patches and later
subspace interventions.  Stable singular subspaces of this operator generate candidate causal
states, while held-out response blocks and full-model interventions determine identification.

The immediate executable consequence is an all-layer native module atlas on is/was-v15 with
answer-margin and final-residual response blocks, followed by head splitting only in causally live
attention modules.  This directly targets computational specification, cross-boundary grouping,
OOD prediction, extraction, manipulation, composition, and stable identification.  It is not a
compression or rank sweep.

## Explicit model and intervention tensors

The checkpoint is an 18-block decoder transformer with residual width `d=1152`, nine attention
heads per block, head/value width `p=128`, and bilinear-MLP hidden width `h=4608`.  For tokens
`x_1,...,x_T`, the residual stream is `X_l in R^{T x d}`.  Ignoring normalization notation, a block
has the contraction graph

\[
 X_l \longrightarrow X_l+A_l(N_lX_l)
 \longrightarrow X_{l+1}=X_l+A_l+M_l(N'_l(X_l+A_l)).
\]

For head `j`, queries, keys, second queries/keys, and values are contractions of the normalized
residual with `d x p` weight slices; their data-dependent attention weights include softmax.  The
concatenated nine-head output contracts through a `d x d` projection.  The MLP is explicitly

\[
 M_l(x)=D_l[(L_lx)\odot(R_lx)],\qquad
 L_l,R_l\in\mathbb R^{h\times d},\quad D_l\in\mathbb R^{d\times h}.
\]

It is degree two in its normalized local input.  The complete network is not a polynomial because
RMS normalization, attention softmax, and the output nonlinearity remain; even if those were frozen,
composition across 18 bilinear blocks raises degree rapidly.  Embedding/output tying and the native
block parameters remain fixed.  Legal residual-basis changes induce inverse gauges on adjacent
weights; source bases `Q -> QG` and coordinates `z -> G^T z` have an orthogonal gauge, and individual
singular vectors are not identifiable inside degenerate response blocks.  Complete physical module
responses and the response operator itself are gauge-independent up to output-coordinate changes.

For construction environment `e`, paired row `i`, and causal action `a`, let the native intervention
replace a specified complete response or subspace response in the recipient trajectory.  Define

\[
 \mathcal R_{(e,i,o),a}
   = o(F(x^{\mathrm{base}}_{e,i};a\leftarrow
             x^{\mathrm{donor}}_{e,i}))-o(F(x^{\mathrm{base}}_{e,i})).
\]

The output/test index `o` is block-structured and includes at minimum: (1) answer-minus-foil margin;
(2) the centered final residual in `R^1152`; (3) centered vocabulary logits; and (4) registered
intermediate readers such as L15 H5/H1 responses.  Each block is whitened only from training rows,
and the approximation norm is a maximum over held-out construction and reader blocks of normalized
vector error, with signed behavior and hard target-retention constraints reported separately.
Scalar averaging across the blocks is forbidden as an identification criterion.

The existing H3 DAS action has a unit vector `u in R^128` and rank-one projector `P=uu^T`; it has
128 fitted parameters before the unit-norm quotient.  More generally, under a local linearization
at environment `e`,

\[
 \mathcal R_e(P)\approx J_e\,\mathrm{vec}_{sym}(P),
\]

where `J_e` is an empirically queried causal Jacobian and `P` is a symmetric PSD projector.  The
linearized parameter count is `p(p+1)/2=8256`, reduced by rank and orthogonal gauge only after the
response data justify that restriction.  Inputs are prospective capability-qualified paired texts;
outputs to preserve are all response blocks plus unrelated-control answers.  Literal price records
model forwards, stored response scalars, intervention edges, and—only for an adopted executor—the
weights/operations needed to compute the action.  The current all-module screen costs about 60
native forwards over 32 rows and stores `36 x 32 x (1+1152)` primary response numbers before optional
reader/logit blocks; it is identification machinery, not yet a storage saving.

## Exact neighboring results and assumption audit

### Block Hankel and Ho–Kalman realization

For an LTI system `s_{t+1}=As_t+Bu_t`, `y_t=Cs_t`, Markov parameters are
`G_k=CA^{k-1}B`, and a block Hankel matrix factors as `H_{r,c}=O_r C_c`.  Under finite-dimensional,
time-invariant, controllable, and observable dynamics, its rank is the minimal state order; a
Ho–Kalman-style factorization recovers a realization up to similarity.  Sarkar, Rakhlin, and Dahleh
construct a noisy finite Hankel-like estimator, select order from data, and recover an approximate
realization with finite-time guarantees for stable LTI systems
([JMLR 2021](https://www.jmlr.org/papers/v22/19-725.html)).

The mapping here is: causal actions are inputs, registered downstream tests are outputs, and depth-
ordered module boundaries resemble time.  But the exact theorem does **not** solve Theseus: the
transition changes by layer, attention is input-dependent, interventions are finite replacements
rather than additive LTI impulses, text rows are not one stationary trajectory, and no current
controllability/observability or stability premise holds.  Ho–Kalman is therefore an algorithmic
restriction for a locally linear, depth-tied toy model, not a neural-circuit identification theorem.

### Weighted automata and Hankel tensors

For a rational series `f(uv)`, the prefix/suffix Hankel matrix has finite rank equal to the minimal
weighted-automaton state dimension.  Rabusseau, Li, and Precup prove an expressive equivalence
between weighted finite automata and linear second-order RNNs and recover a linear 2-RNN from
low-rank Hankel tensor blocks under rank/sufficiency assumptions
([AISTATS 2019](https://proceedings.mlr.press/v89/rabusseau19a.html)).  Denis, Gybels, and Habrard
give dimension-free concentration bounds for empirical Hankel matrices used in spectral learning
([ICML 2014](https://proceedings.mlr.press/v32/denis14.html)).

Our construction/readout blocks play the roles of prefixes/suffixes, and the bilinear MLP makes the
2-RNN/tensor analogy concrete.  The assumptions fail globally: the transformer is neither a linear
2-RNN nor known to compute a rational series; our finite text set is not closed under prefix/suffix
concatenation; interventions change hidden trajectories; and response samples are deliberately
stratified rather than iid strings.  Consequently Hankel rank can summarize the queried operator,
but cannot certify a minimal neural program without held-out causal closure.

### Predictive-state causal quotient and perturbation

Predictive-state representations replace latent coordinates by predictions of observable future
tests.  The exact operational analogue here is the quotient

\[
 a\sim b \quad\Longleftrightarrow\quad
 \mathcal R_{q,a}=\mathcal R_{q,b}\ \text{for every admissible downstream test }q.
\]

This removes arbitrary activation gauge and matches the desired notion that two tensor directions
are the same circuit variable only when all consumers treat them equivalently.  The current finite
test family is not proven complete, as the `.928` response/`.399` behavior mismatch demonstrates;
the quotient is only as good as its tests.  Adding final residual, vocabulary, and margin blocks is
therefore a mathematical necessity rather than extra ceremony.

Wedin's singular-subspace perturbation result bounds response-subspace rotation by operator
perturbation divided by the relevant singular gap
([BIT 1972](https://doi.org/10.1007/BF01932678)).  Applied exactly to empirical train/test response
matrices, it licenses naming a response block only when the construction perturbation is small
relative to the retained/discarded gap.  It does not make unstable singular vectors identifiable;
near-degenerate directions must be reported as a block.

## Executable consequence with opposing predictions

Build `R_train` and `R_holdout` with identical intervention columns and separately normalized
behavior, final-residual, vocabulary, and intermediate-reader rows.  The first intervention basis is
all 36 complete native modules plus cumulative depth prefixes on is/was-v15.  Record exact base-self
and all-module-to-donor closure.  Then:

1. rank sites by held-out behavioral and final-residual response, never by activation variance;
2. identify the earliest cumulative prefix that restores at least `.75` signed behavior;
3. split only causally live attention sites into nine heads and only live MLPs into response modes;
4. compare singular subspaces across A1/A2 and future construction blocks with Wedin gap reports;
5. require fitted response modes to predict a wholly held-out reader block and full-model behavior;
6. pull an identified response covector through exact weights to rank writers/readers, then verify
   those ranks by complete causal patches.

Opposing outcomes are sharp.  A dominant new module or head supports localized construction routing;
many small singleton columns plus a successful prefix supports a distributive graph and licenses an
executed greedy composition; strong old-reader response with weak final-residual/behavior response
confirms observation-map failure; instability across A1/A2 despite a spectral gap rejects a shared
causal state; and stable vector prediction with successful selective swaps promotes the finite
causal quotient toward identification.  Failure of all proper prefixes while the all-module arm
replays donor means interaction/order, not a missing singleton, and requires factorial grouping.

This empirical route currently dominates an attempted exact Hankel realization: it needs tens of
forwards, makes fewer false structural assumptions, and directly resolves the live graph-versus-
observation uncertainty.  The spectral theorem becomes useful after the response matrix exists,
as a stability and minimal-realization diagnostic—not before it as a substitute for intervention.
