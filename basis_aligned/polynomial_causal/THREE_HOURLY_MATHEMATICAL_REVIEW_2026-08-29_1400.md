# Three-hour mathematical review — 2026-08-29 14:00 UTC

## Starting point from the actual model evidence

The new fact that should control the mathematics is not another local low-rank fit.
The registered four-head copy intervention has copy CE effect `0.44870` nat and
specificity `0.46352` nat, but its off-target effect is `0.02441` nat, above the
frozen `0.01` collateral budget. The four singleton copy effects sum to only
`0.10705` nat. Thus the useful computation is strongly joint, while the crude
position-mean replacement removes shared non-copy work too.

The current context-free whole-model compiler also remains at roughly 14% top-1
accuracy against 39–42% live. MLP-heavy table allocation and a larger fallback map
give small, useful price/long-tail improvements, but they do not name the missing
contextual computation. Strict ledgers remain 36/36 structural sites,
5.348245316% certified storage removal, 10.923302467% named causal CE, 4.72714 nat
unexplained, and 0/68 terminal actions.

## Ranked mathematical move 1: Boolean causal interaction decomposition

### Exact object

Let the four selected heads be the ground set

$$
F=\{\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4}\}.
$$

For each copy, matched-negative, and off-target cell, define the causal set function

$$
v_c(S)=\mathrm{CE}_c(\text{replace exactly heads }S)-\mathrm{CE}_c(\text{native}),
\qquad S\subseteq F.
$$

Because the model fixes the temporal order of layers, every subset specifies one
unambiguous sequential intervention. The unique Boolean-lattice/Möbius expansion is

$$
m_c(T)=\sum_{S\subseteq T}(-1)^{|T|-|S|}v_c(S),
\qquad
v_c(S)=\sum_{T\subseteq S}m_c(T).
$$

This is the exact discrete analogue of decomposing a polynomial into main effects
and interactions. Shapley–Taylor indices provide an efficient attribution that
preserves total effect while assigning higher-order interactions up to a chosen
order; the paper relates this to the Taylor expansion of the multilinear extension
of a set function ([Sundararajan, Dhamdhere, and Agarwal, 2020](https://proceedings.mlr.press/v119/sundararajan20a.html)).

### What it predicts beyond reconstruction

If copy effect is concentrated in one or two sparse interaction coefficients while
off-target damage is mostly additive or belongs to different coefficients, we gain a
principled circuit boundary. We can then extract or suppress the interaction rather
than deleting four entire head outputs. The decomposition composes naturally: each
coefficient is an intervention contrast in the same residual-stream currency.

### Assumptions that may fail

- The position-mean baseline must define a scientifically meaningful intervention.
- Interaction coefficients are baseline-dependent; they are not unique semantic
  properties of the raw weights.
- A sparse subset decomposition may still fail because the true component is
  conditional on token context rather than head identity alone.
- The already exposed E4 role cannot be reused to promote a newly chosen subset.

### Cheapest falsifying experiment

Measure all 16 subsets on one new, prospectively frozen role with the same shared
native baseline. Ten subset arms are missing from E4. Compute document-paired Möbius
coefficients and simultaneous bounds separately for copy, matched-negative, and
off-target cells. Falsify the sparse-interaction hypothesis if the effect is diffuse
across many coefficients or the low-order truncation does not predict held-out subset
effects.

### CPU analysis executed now

The saved ledger permits one contrast without new model calls:

$$
e_c=v_c(F)-\sum_{i\in F}v_c(\{i\}).
$$

This is **not** a four-way Möbius coefficient because pairs and triples are missing.
It is only a test of additivity. On 192 paired documents with 10,000 deterministic
bootstrap draws:

| Contrast | Point | Simultaneous lower | Simultaneous upper |
|---|---:|---:|---:|
| Copy-positive excess | 0.34165 | 0.20810 | 0.49534 |
| Matched-negative excess | 0.02372 | -0.10983 | 0.17741 |
| Off-target excess | 0.01638 | -0.11716 | 0.17008 |
| Specificity excess | 0.31793 | 0.18438 | 0.47162 |

The joint positive effect is 4.1916 times the singleton sum. Additivity therefore
misses most of the copy effect and specificity, robustly at the document level. The
result is explicitly post-hoc/descriptive and does not locate the interaction order.

## Ranked mathematical move 2: approximate causal abstraction with a conditional replacement

### Exact object

The low-level variables are the four heads' live value/write tensors and the
downstream residual stream. Define an executable macrostate

$$
Z_t=(\text{nearest prior query exists},\text{query distance},
\text{query token},\text{candidate successor relation},\text{position features}),
$$

then learn a small replacement program $R(Z_t)$ for only the copy-dependent component.
The high-level program is useful only if its allowed interventions commute
approximately with low-level interventions: applying “remove copy” before or after
the abstraction should give nearly the same downstream distribution. This is the
operational content of approximate causal abstraction, which explicitly measures the
discrepancy between low- and high-level causal models rather than demanding exact
identity ([Beckers, Eberhardt, and Halpern, 2020](https://proceedings.mlr.press/v115/beckers20a.html)).

Conditional expectation $E[H\mid Z]$ is the $L^2$-optimal function of $Z$, but that
fact is only a proposal generator. Acceptance must use intervention CE/KL,
specificity, collateral, and OOD transport—not activation MSE.

### What it predicts beyond reconstruction

A valid macrostate should retain the four-head copy effect while leaving the
complementary/non-copy contribution native or cheaply reconstructed. It gives an
executable selective-removal instruction and a direct OOD prediction: contexts with
the same $Z$ should have similar intervention responses even when surface tokens or
document domain change.

### Assumptions that may fail

- The hand-defined $Z$ may omit a hidden contextual parent.
- Conditional means can erase multimodal structure or nonlinear interactions.
- A gated intervention may remove a behavior without extracting a standalone
  generative program.
- RMSNorm and later attention may make small local errors large and context-dependent.

### Cheapest falsifying experiment

On a new discovery role, compare three frozen interventions: unconditional
position-mean replacement, online copy-predicate-gated replacement, and a matched
random gate with the same activation rate. If gating does not reduce off-target CE
below 0.01 while retaining at least half the registered copy effect, the proposed
macrostate is insufficient. A successful pilot still requires a separate natural/OOD
replication before extraction claims.

## Ranked mathematical move 3: downstream-Fisher active subspace and shared dictionary

### Exact object

Let $h$ concatenate the selected head outputs (or the MLP0 product code) at an
interface, $z=G(h)$ be downstream logits, $J=\partial z/\partial h$, and

$$
F(p)=\operatorname{diag}(p)-pp^\top
$$

be the categorical Fisher matrix. The second-order local behavioral cost is

$$
\mathrm{KL}(p(z)\Vert p(z+J\delta h))
\approx \tfrac12\delta h^\top J^\top FJ\delta h.
$$

Average $J^\top FJ$ over documents/cells and diagonalize it, or solve the generalized
eigenproblem against activation covariance. A shared basis $U$ is then chosen in this
downstream metric and each head/site receives sparse coefficients relative to $U$.
This is a causal-metric simultaneous factorization, not PCA/HOSVD of raw writes.
Active-subspace theory uses eigenvectors of average gradient outer products and gives
ridge-approximation error bounds under distributional/Poincaré assumptions
([Constantine, Dow, and Wang, 2014](https://epubs.siam.org/doi/abs/10.1137/130916138)).

### What it predicts beyond reconstruction

The eigenvalue tail predicts the minimum dimension needed for local CE/KL and finite
edit transport. A basis shared by several heads or MLP sites gives an executable
dictionary whose price is basis storage plus sparse per-site coefficients. Unlike raw
HOSVD, its discarded directions are certified as downstream-insensitive to second
order on the measured distribution.

### Assumptions that may fail

- The Fisher/Jacobian approximation is local; E4's finite mean intervention may leave
  that regime.
- The average metric can hide rare but important directions.
- Poincaré/error-bound assumptions need not hold for discrete, multimodal language
  states.
- Gauge whitening can be ill-conditioned, and a shared basis may not survive OOD.

### Cheapest falsifying experiment

Use randomized JVP/VJP sketches on 32 discovery documents to estimate the top
eigenspace. At ranks 8, 16, and 32, compare held-out centered-logit response, KL,
finite edit transport, and collateral against activation PCA and Haar controls. Stop
if eigengaps are unstable under document doubling or the causal basis does not beat
PCA at matched rank and executable cost.

## Ideas reconsidered and pruned for now

- **Raw tensor rank, CP/Tucker/HOSVD, and norm-minimization before HOSVD:** useful
  canonical proposal generators, but already too close to the local reconstruction
  objective that Family F showed can select the worse downstream decoder. HOSVD is
  rank-revealing, not generally best multilinear-rank approximation. Gauge
  canonicalization does not by itself separate copy from collateral.
- **Full tensor-network minimal canonical form/invariant theory:** mathematically
  attractive—the PEPS minimal canonical form identifies tensors equivalent up to
  gauge-orbit closure and equality across geometries
  ([Acuaviva et al., 2022](https://arxiv.org/abs/2209.14358))—but bilin18 is a fixed
  nonlinear residual/RMSNorm computation, not the paper's family of PEPS contractions.
  Apply this only after isolating a polynomial subnetwork whose gauge group and output
  equivalence are explicit.
- **Hankel/minimal realization:** finite Hankel rank equals minimal weighted-automaton
  state dimension for rational series, and spectral rank factorization recovers a
  realization ([Arrivault et al., 2016](https://proceedings.mlr.press/v57/arrivault16.html)).
  This is promising for a restricted copy language, but current natural-text logits
  are neither known rational nor covered by a complete prefix/suffix basis. It ranks
  after the 16-subset interaction cube.
- **MDL/prequential coding:** retain as the final model-selection currency. It can
  decide whether a shared dictionary pays for itself on fresh data, but it will not
  discover which four-head interaction is causal.
- **Information bottleneck:** distribution-dependent mutual information can discard
  intervention-relevant rare states and has no automatic composability guarantee.
- **Unconstrained sparse program synthesis/SAE on weights:** too many equivalent
  programs and gauge choices. Require a causal interface and behavior-based score
  first.
- **More allocation or map-rank sweeps:** current jobs already price these levers;
  their gains are engineering refinements, not an explanation of the 4.72714-nat gap.

## Operational priority

1. Freeze the full 16-subset four-head interaction experiment and its new role.
2. In parallel, finish the already-designed native-Down behavioral-port measurement
   validator/row audit; it tests a different interface and should not wait on copy.
3. If the interaction cube is sparse, build the conditional causal abstraction and
   test selective removal. If it is diffuse, skip semantic subset naming and move
   directly to the downstream-Fisher shared basis.

