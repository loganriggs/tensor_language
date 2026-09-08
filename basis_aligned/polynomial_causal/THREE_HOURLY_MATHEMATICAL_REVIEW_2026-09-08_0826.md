# Three-hour mathematical tensor-network review — 2026-09-08 08:26 UTC

## Exact current object

Bilin18 has residual width `d=1152`, `L=18` blocks, `H=9` heads of width `p=128`,
bilinear-MLP width `4608`, and vocabulary size `50304`.  For batch row `b`, destination token
`i`, source token `j`, head `h`, and value coordinate `c`, L11 attention uses normalized rotary
factors

\[
q,q',k,k'\in\mathbb R^{B\times T\times H\times128},\qquad
v\in\mathbb R^{B\times T\times H\times128},
\]

and the causal unnormalized routing tensor

\[
A^h_{bij}=1_{j\le i}
 \frac{\langle\widehat q_{bih},\widehat k_{bjh}\rangle}{128}
 \frac{\langle\widehat q'_{bih},\widehat k'_{bjh}\rangle}{128}.
\]

Before RMS normalization this score is degree four in residual coordinates and its contraction
with `v` is degree five.  RMS normalization makes the deployed function rational/algebraic rather
than globally polynomial.  The downstream graph contains further normalized bilinear attention,
quadratic `Down(Left(x)*Right(x))` MLPs, tied unembedding, and elementwise soft cap
`30 tanh(logit/30)`.

For role `r` in `{temporal,iswas}`, let `R_r(b)` be the already locked semantic source positions:
bridge for temporal and postcue for is-was.  Let `P_r` be the corresponding row-dependent diagonal
mask.  The exact L11H3 controlled value mediator for an arbitrary induced value change `Delta V`
is

\[
M_r(\Delta V)=(1-\lambda_{11})A^3_0P_r\Delta V^3
 \in\mathbb R^{B\times T\times128},
\]

where `A_0` is recipient-native routing.  The executed source-term extraction already proves this
tensor equals the independently observed L11 value-patch preprojection delta within `9.54e-6`,
and direct installation reproduces the selected logits within `8.59e-6`.

The pending greedy experiment chooses a role-specific nested prefix `S_r` from five physical
earlier-head sites.  Its full writer intervention produces logits `Y(W_{S_r})`, L11 value
`V^{W,S_r}`, and effect vector

\[
e^W_r=m_r(Y(W_{S_r}))-m_r(Y_0)\in\mathbb R^{128}.
\]

The frozen mediation successor computes

\[
\Delta V_r=P_r(V^{W,S_r}-V^0),\qquad
e^M_r=m_r\!\left(Y_0\;\text{with}\;z^{11,3}+M_r(\Delta V_r)\right)-m_r(Y_0).
\]

It measures projection `<eM,eW>/<eW,eW>`, cosine, direction agreement, and relative residual
`||eM-eW||/||eW||` on original HOLDOUT and both OOD phases.  This is an exact controlled
intervention through one mediator and a nonlinear downstream suffix.  It is not an algebraic
claim that the entire writer effect is linear in `Delta V`: the writer can also alter L11 routing,
values outside `R_r`, other heads, and later residual paths.

## Symmetries, allowed inputs, and literal price

The q/k and q'/k' dot products are invariant under paired orthogonal head-coordinate changes;
the value/output interface is invariant when a value-basis change is cancelled by the inverse
output projection.  Per-head RMS normalization does not permit arbitrary `GL(128)` gauges.  The
stored 128-vector mediator tensor is coordinate dependent, while its installed residual effect is
invariant under the paired checkpoint reparameterization.  The current experiment fixes the
physical checkpoint gauge and byte-hashes every authority.

Allowed inputs are exactly 128 original and 128 fresh-OOD endpoints with involutive command donor
pairs.  Original FIT can select the greedy prefix but cannot support the mediation claim;
original HOLDOUT and OOD FIT/HOLDOUT are validation.  Outputs are canonical fixed-orientation
`will-had` and `is-was` margins, never correct-minus-foil coordinates that reverse with the bit.

The greedy test costs 26 model forwards, 3,328 sequence evaluations, and zero fitted parameters.
The mediation test costs 18 forwards: for each population one native capture and, for each role,
one writer arm, one live L11 command-donor reference, one value-mediator arm, and one direct-formula
arm.  That is 2,304 sequence evaluations, 4,608 scored token positions, zero backwards, zero
updates, and zero fit parameters.  Conditional on already available routing scalars, evaluating
the explicit source term costs `128` multiply-accumulates per destination/source pair; producing
those routing scalars still requires the two 128-dimensional q/k dot products and native context.
No native block or parameter is yet eliminated.

## Theorem and algorithm mappings

### 1. Controlled mediation is identifiable here; natural mediation language is unnecessary

Robins and Greenland show that randomizing only an exposure generally does not identify direct
and indirect effects; separation needs additional assumptions, while a controllable intervention
on the intermediate can support a controlled decomposition ([Epidemiology 1992,
doi:10.1097/00001648-199203000-00013](https://pubmed.ncbi.nlm.nih.gov/1576220/)).  Pearl likewise
distinguishes total, controlled direct, and path-specific effects in nonlinear models
([Direct and Indirect Effects](https://arxiv.org/abs/1301.2300)).

**Object mapping.** Exposure is the exact earlier-head donor patch `W_{S_r}`; mediator is the
source-local L11H3 value tensor; outcome is the fixed command margin.  Unlike observational
mediation, every model variable is observed and the mediator is directly set by a hook.  We do
not need exchangeability, no-unmeasured-confounding, or cross-world assumptions to identify the
specific controlled intervention effect `eM`.  We also do not obtain a unique natural indirect
effect: the hybrid state “native routing with writer-induced value” is an engineered controlled
counterfactual.

**Executable consequence.** Keep the separate writer, value-mediator, and direct-formula arms.
Require formula/value equality before interpreting recovery, and call the result a controlled
L11-value mediation fraction.  If recovery fails, preserve the residual as a parallel causal
route; do not optimize a latent subspace until the residual has its own intervention semantics.
This is exactly the already built 18-forward successor.

### 2. Classical greedy guarantees do not apply to the frozen weight order

Nemhauser, Wolsey, and Fisher analyze cardinality-constrained greedy maximization for monotone
submodular set functions ([Mathematical Programming 14, 1978](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf)).
Das and Kempe extend useful guarantees to sparse linear prediction through a positive
submodularity ratio and sparse covariance eigenvalues ([ICML 2011](https://icml.cc/2011/papers/542_icmlpaper.pdf)).

**Object mapping.** For a head subset `S`, a natural fidelity utility would be

\[
F_r(S)=1-\frac{\|e_{r,S}-g_r\|_2^2}{\|g_r\|_2^2},
\]

where `g_r` is the L11 value-parent effect.  The current greedy order, however, is sorted by
trace-normalized physical weight reach, not by the marginal gain of `F`.  Moreover, the is-was
top-five causal arm has recovery `2.30-2.91`; adding writers can move past the target and decrease
`F`.  Thus monotonicity is already contradicted.  The measured top-five singleton-sum error
`.17425` establishes approximate additivity only at one union, not submodularity on all subsets,
and supplies no positive submodularity-ratio certificate.

**Executable consequence.** A passing nested prefix is prospective evidence for that frozen
weight order and a literal edge-count reduction, not an approximation theorem or globally minimal
subset.  A failure falsifies the order without proving that no compact subset exists.  Future
reusable causal set-function runners should retain rowwise effect vectors so a five-element
submodularity ratio or exact 32-subset certificate can be computed when scientifically licensed;
scalar recovery summaries are insufficient.  Do not change the live greedy experiment or inspect
validation to choose another prefix.

### 3. Tensor-train compression solves a different object

TT-SVD constructs tensor-train representations from low-rank unfoldings and supports stable
rounding and tensor operations ([Oseledets, SIAM J. Scientific Computing 2011](https://epubs.siam.org/doi/10.1137/090752286)).

**Object mapping.** One could arrange stored query tensors as task x population x row x value
coordinate or factor the routing/value contraction across token indices.  TT ranks would bound
storage approximation error under unfolding norms.  They would not identify the semantic source,
prove causal writer use, preserve downstream command margins under installation, or remove the
native routing background.  The input-dependent normalization also prevents treating one fixed
coefficient tensor as the full deployed map.

**Decision.** TT is deferred until the writer edge and joined H4 program pass.  At that point it
may be a matched engineering control for literal storage/compute; today it would be another
reconstruction/compression detour.

### 4. Hankel-rank minimal realization is not yet applicable to the deployed interface

Carlyle and Paz characterize finite-rank string functions by finite-dimensional weighted
sequential realizations ([Journal of Computer and System Sciences 1971](https://www.sciencedirect.com/science/article/pii/S0022000071800053)).
The relevant Hankel matrix indexes prefixes and suffixes, and its rank gives a minimal linear
state dimension under the theorem's assumptions.

**Object mapping and violation.** A fully compiled controlled-language circuit could define a
string-to-margin function and admit a finite Hankel test.  The current L11 interface instead takes
native continuous q/k context and writer-induced values from the transformer; it is not a closed
linear weighted automaton over tokens.  A finite benchmark table would certify only that table,
not free-form realization.

**Decision.** Once the native context dependency is compiled or explicitly bounded, build a
prefix/suffix Hankel matrix of the executable circuit and use rank as a lower-bound/minimal-state
certificate.  It cannot replace the present causal mediation experiment.

## Mathematical decision and immediate action

The controlled-mediation mapping dominates: it exactly matches the intervention access available
inside the transformer, avoids unidentifiable natural-effect language, and produces an opposing
prediction at the explicit tensor edge.  The submodularity audit narrows the greedy claim and
prevents a passing weight prefix from being mislabeled globally optimal.  TT and Hankel methods
remain potentially valuable only after the causal program boundary is closed.

The immediate executable consequence is already underway: retain the hash-bound greedy job behind
the verified-live managed predecessor; on a valid result, bind only its SHA and selected identities
into the frozen 18-forward mediation runner.  A full mediation pass licenses joining this exact
edge with H4 composition/removal.  A partial result must quantify the residual and redirect to an
operational parallel-route causal screen, not DAS regularization, rank reduction, or threshold
retuning.
