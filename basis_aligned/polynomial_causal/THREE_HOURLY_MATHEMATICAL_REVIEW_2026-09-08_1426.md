# Three-hour mathematical tensor-network review — 2026-09-08 14:26 UTC

## Exact current object

Bilin18 has `L=18` blocks, residual width `d=1152`, `H=9` attention heads of width
`d_h=128`, bilinear-MLP hidden width `p=4608`, and vocabulary `V=50304`. The token/state
tensor is `X[b,t,r]`. Embedding and unembedding are tied. Each attention module contracts
normalized residuals through Q/K/Q2/K2 score factors, a value factor, and an output projection;
each MLP contracts

\[
m^l_{btr}=\sum_{u=1}^{4608}D^l_{ru}
  \left(\sum_s L^l_{us}\,\bar X^l_{bts}\right)
  \left(\sum_q R^l_{uq}\,\bar X^l_{btq}\right),
\]

where `bar X` is the deployed RMS-normalized input. The native transformer is not a polynomial
tensor network because RMS normalization, normalized attention factors, and the final tanh softcap
remain live. This review therefore does not apply a polynomial-decomposition theorem to the
checkpoint weights.

The intervention now in flight defines a different exact finite object. Let

\[
F_{s,z,r,c}\in\mathbb R,
\]

where `s in {0,1}` is the block-10 residual source state (native or selected writer),
`z in {0,1}^7` says which P7 modules recompute live, `r=1,...,128` indexes the fixed combined
OOD FIT/HOLDOUT rows, and `c` indexes the is-was target margin or the temporal collateral margin.
The seven mask players are stored as

`(A11,M11,M12,M15,M13,M16,M10)`;

their physical execution order is `M10,A11,M11,M12,M13,M15,M16`. A zero mask bit clamps that
module's output to its native row/position tensor at the module's own stage. A one bit leaves the
module live. All non-P7 modules remain live. The source-effect game for row `r` is

\[
E_r(S)=F_{1,S,r,\mathrm{iswas}}-F_{0,S,r,\mathrm{iswas}}.
\]

`E(empty)` is the direct or unlisted-background effect. The allocated mediated increment is
`E(N)-E(empty)`. The registered behavioral scalar for any vector `v` is its signed credit against
the full effect `g=E(N)` on one frozen phase,

\[
\rho_g(v)=\frac{\langle v,g\rangle}{\langle g,g\rangle}.
\]

This is a directed effect coordinate, not cosine similarity and not a probability.

## Indices, contraction graph, symmetries, norms, and literal price

The controlled contraction graph is

\[
X^{10}_{s}\longrightarrow
M10\longrightarrow A11\longrightarrow M11\longrightarrow M12\longrightarrow
M13\longrightarrow M15\longrightarrow M16\longrightarrow
\operatorname{RMSNorm}\longrightarrow W_U\longrightarrow \tanh\longrightarrow F,
\]

with every omitted native suffix module still present as live background. The intervention tensor
has shape `2 x 128 x 128 x 2`; only aggregate target/collateral vectors and transforms are retained
in the result. The unique multilinear extension of the target game has degree at most seven even
though the underlying model is non-polynomial.

Relabeling the seven players while applying the same permutation to mask bits, Möbius coefficients,
and reported labels leaves the game invariant. The hash-locked tuple is therefore bookkeeping, not
a semantic order. The physical checkpoint also admits paired attention-coordinate gauges,
V/O inverse gauges, MLP Left/Right reciprocal scaling, hidden permutations, and a globally
consistent residual change of basis. Module clamping and the causal graph are operational in the
deployed checkpoint gauge; an attribution to a native module is not invariant to an arbitrary
cross-module reparameterization. Exact checkpoint contractions followed by physical interchange
are required before a weight edge can be called identified.

Allowed inputs are the immutable 64-row OOD FIT and 64-row OOD HOLDOUT banks already used by the
parent interface. No row, subset, player, sign, or threshold is fitted. Outputs are the signed
answer/foil margin vectors and temporal command-gold collateral. Exact-transform error is measured
in maximum absolute float64 error (`<=1e-8`); executed no-op and parent replay use `<=1e-5`;
route and pair nominations use absolute signed credit `>=.03` with the same sign on both phases.
The literal price is one checkpoint load, 258 forwards (two captures plus `2*2^7` game arms),
33,024 sequence evaluations, 66,048 scored positions, zero backwards, zero updates, and zero fitted
parameters. It removes no model weights yet.

## Exact theorem and algorithm mappings

### Boolean-lattice Möbius inversion exactly solves the finite intervention table

For every `T subseteq N`, define the vector dividend

\[
d(T)=\sum_{U\subseteq T}(-1)^{|T|-|U|}E(U).
\]

Then `E(S)=sum_{T subseteq S} d(T)` uniquely. This is ordinary incidence-algebra inversion on the
Boolean lattice, the finite-poset object formalized by Rota
([Rota 1964](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf)). The fast in-place subset
transform costs `O(n 2^n B)` additions for `n=7`, `B=128` row values and needs `O(2^n B)` memory.
Its assumptions are only a complete finite table and fixed player identities; both hold. It has
exact uniqueness and reconstruction, but it says nothing about whether a player is a semantic unit
or whether an interaction is a directed network edge.

The Shapley allocation of the mediated increment follows directly from the dividends:

\[
\phi_i=\sum_{T\ni i}\frac{d(T)}{|T|},\qquad
E(N)-E(\varnothing)=\sum_i\phi_i.
\]

The implementation checks this identity per row to `1e-8`. This is the exact algorithm for the
current seven-player object; approximation is unnecessary.

### Owen's multilinear extension is an equivalent exact representation

Owen defines the unique function on the unit cube that is linear in each coordinate and agrees
with a game at every Boolean corner
([Owen 1972](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.18.5.64)). For this game,

\[
f(x)=\sum_{T\subseteq N}d(T)\prod_{i\in T}x_i.
\]

The object mapping is exact: corner `x=1_S` is the executed coalition `E(S)`, and the coefficients
are the Möbius dividends above. Shapley values can be obtained from diagonal derivative integrals.
The guarantee is exact equality and unique multilinear interpolation once all corners are known.
It neither reduces the `2^7` oracle calls needed for an arbitrary black-box game nor identifies
which residual coordinates a module reads or writes. It is useful here as a correctness proof and
as a route to continuous dose diagnostics later, not as a causal-edge theorem.

### Grabisch-Roubens interaction is an exact index, not an edge certificate

The implemented pair index is

\[
I_{ij}=\sum_{T\supseteq\{i,j\}}\frac{d(T)}{|T|-1}.
\]

Interaction indices generalize Shapley-style allocation and aggregate higher-order dividends
([Grabisch and Roubens 1999](https://doi.org/10.1007/s001820050125)); the related interaction
transform is linear and invertible, with singleton terms equal to Shapley values
([Denneberg and Grabisch 1999](https://www.sciencedirect.com/science/article/abs/pii/S0020025599000997)).
The mapping requires a fixed cooperative game and complete coalition values, which hold. The index
is uniquely determined by the chosen axioms/convention and costs `O(n^2 2^n B)` by direct dividend
aggregation. It deliberately includes every higher-order coalition containing the pair. Therefore
a negative value may mean redundancy or cancellation and a positive value complementarity, but
neither sign proves a physical `i -> j` edge. Four-corner inspection and a directed intervention
remain mandatory.

### Tensor-network Shapley surrogates solve the large-player approximation, not this game

TN-SHAP-G learns a graph-aligned multilinear tensor-network surrogate from a subset of oracle
queries, then computes Shapley values and interactions exactly for that surrogate
([Heidari and Rabusseau 2026](https://arxiv.org/abs/2606.01540)). The proposed mapping would treat
our seven module bits as graph nodes and `E(S)` as the black-box corner value. Its tractability
depends on a low-bond-dimension approximation to the complete coalition tensor; attribution error
then depends on surrogate fidelity. No theorem in that mapping identifies a native module edge,
and held-out mean-square fit would not control the signed `.03` threshold without a stronger
uniform error bound.

For `n=7`, exact enumeration needs only 256 intervention arms and avoids training, restart,
regularization, post-selection, and approximation error. The managed GPU bottleneck is another
job ahead of ours, not exponential cost. TN-SHAP-G is therefore dominated for the current object.
It becomes relevant only when a prospectively fixed player set is too large to enumerate and the
surrogate is validated on sealed coalitions with a bound strong enough to preserve circuit verdicts.

### Weight tensor decomposition still does not solve the live nonlinear graph

TT-SVD, hierarchical Tucker, or polynomial tensor decomposition can compress a stored coefficient
tensor or an exactly polynomial restriction. The native suffix contains live RMS normalization,
attention, and bilinear MLP recurrence, so a fixed coefficient tensor does not represent this
intervention map without changing the instrument. Low checkpoint rank or activation reconstruction
would nominate a basis but would not establish OOD source mediation, selective manipulation, or
composition. These algorithms remain downstream engineering after operational pieces are identified.

## Executable consequence and opposing predictions

The exact finite game is the correct algorithm now. Its outcome selects one of two physical tests:

1. If a module `j` has stable same-sign Shapley credit, freeze an edge-specific causal mediation
   factorial on a sealed row bank. Capture `j`'s normalized reader input and output with the source
   absent and present. In the source-present arm compare: native live `j`; reader-input replacement
   by the source-absent tensor; that reader replacement plus restoration of the captured
   source-present module output; and a matched irrelevant-coordinate/control-module replacement.
   A directed `X10 -> j` route predicts that reader replacement removes at least the registered
   game credit, source-present output restoration selectively rescues it, and the control does not.
   Failure of loss or rescue kills the edge claim even if Shapley is large. For attention, split
   the reader intervention prospectively over Q/K/Q2/K2/V only after complete-reader mediation;
   for an MLP, split Left/Right only after complete normalized-input mediation.
2. If no module is stable while `E(empty)` remains material, P7 is not the identity mediator at
   this grain. Freeze a cheap causal-order-filtered atlas over non-P7 downstream modules using the
   same source-present/source-absent effect and executed no-op controls. This follows the direct
   background route rather than widening P7 or optimizing a subspace on the observed OOD game.

If a pair is stable, the edge test expands to the two nominated reader removals and single/joint
output rescues; a joint-only rescue supports complementarity, while either single rescue supports
redundancy. In all cases weight contractions nominate legal reader tensors only after a passing
operational module test, and a new sealed population is required because this OOD bank selected the
module or pair.

This consequence advances computational specification, within-module splitting, OOD prediction,
selective manipulation, composition, and stable identification. Its evidence is causal loss and
rescue through a registered reader interface. No rank, reconstruction, or variance result can
substitute for those measurements.

## Mathematical decision

Continue the exact effect game and do not introduce a surrogate. Möbius inversion, Owen's
multilinear extension, and Grabisch-Roubens interaction exactly solve the attribution algebra but
stop before network identification. The highest-information successor is therefore a physical
reader-input removal/output-rescue factorial for any stable module, or a direct-background module
atlas if the P7 credits are null. Preregister that successor only after the immutable game terminal
selects the branch, then execute it through the managed runner.
