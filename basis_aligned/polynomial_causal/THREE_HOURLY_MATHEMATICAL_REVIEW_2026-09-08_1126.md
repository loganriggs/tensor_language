# Three-hour mathematical tensor-network review — 2026-09-08 11:26 UTC

## Exact current object

Bilin18 has residual width `d=1152`, `L=18` blocks, `H=9` attention heads of width `128`,
bilinear-MLP width `4608`, and vocabulary `50304`. For batch row `b`, token `i`, and suffix block
`l=10,...,17`, write the deployed residual recurrence under externally clamped complete module
writes as

\[
x^{l+1}_{bi}=\lambda_{l,0}x^l_{bi}+\lambda_{l,1}x^0_{bi}
                 +a^l_{bi}+m^l_{bi},
\]

where `a` and `m` are the captured attention and bilinear-MLP outputs. Each native attention write
contains two normalized QK contractions and a value contraction; each native MLP write is
`Down(Left(x) * Right(x))`. Those writes are high-degree rational/algebraic functions of an
unclamped state because RMS normalizations intervene, but after their absolute values are clamped
the recurrence above is affine in the entry state.

Let `x00` be native entry and native module writes, `delta=x10_writer-x10_native`,
`g=product_l lambda[l,0]`, `r=g delta`, and let `m` denote the final-state displacement produced by
writer module outputs with native entry. On the registered causal-prefix positions `P_b`, the four
factorial states obey

\[
X_{00}=x,\quad X_{10}=x+r,\quad X_{01}=x+m,\quad X_{11}=x+r+m.
\]

The exact cumulative result selected seven complete responses
`A11,M11,M12,M15,M13,M16,M10`; its FIT/HOLDOUT recovery is `.5276/.5376`. The full response arm
`X01` is `.4896/.5029`; residual identity `X10` is `.5209/.5088`; `X11` is exactly the writer
effect. The task-margin additivity residual is `.0128/.0148`, so both physical branches matter and
the registered majority threshold happens to separate two near-halves.

The final readout for a row and scored query is

\[
F(x)=M\left(30\tanh\left(W_U\operatorname{RMSNorm}(x)/30\right)\right),
\]

where `W_U` is the tied embedding/unembedding tensor and `M` selects the fixed-orientation is/was
margin. This map is smooth on the observed nonzero states but is rational plus transcendental, not
polynomial. Its four-cell interaction is

\[
I=F(x+r+m)-F(x+r)-F(x+m)+F(x).
\]

The physical-state cross-difference is exactly zero up to deployed rounding; any nonzero `I`
therefore belongs to final normalization/readout curvature, not to an interaction inside the
absolutely clamped suffix.

## Indices, symmetries, allowed inputs, norms, and price

The state tensor is `B x T x 1152`; complete write banks contain sixteen tensors of that shape.
The compact writer is fixed at postcue `L7H7+L9H4`; no coordinate or population is fitted.
Attention Q/K factors admit paired orthogonal head-coordinate gauges and V/O admits paired inverse
changes, while per-head normalization prevents arbitrary `GL(128)` gauges. The raw residual route
is expressed in the physical checkpoint gauge, but its operational add/remove equality is invariant
under any globally consistent reparameterization.

Allowed inputs are 128 original and 128 history-disjoint OOD endpoints, each divided into 64 FIT
and 64 HOLDOUT rows. Interventions are applied only to row-specific causal prefixes `P_b`; outputs
to preserve and compare are the answer/foil logits and margins at each registered query, plus the
temporal command margin as collateral. State error is float32 relative L2 on `P_b`; behavior uses
signed recovery, cosine, direction agreement, and exact selected-logit differences. The direct
experiment costs one checkpoint load, fourteen forwards, 1,792 sequence evaluations, 3,584 scored
positions, no backwards/updates, and no fitted parameters. It removes no native parameter yet.

## Exact theorem/algorithm mappings

### Affine superposition exactly solves the controlled hidden suffix

Repeated substitution gives

\[
X^{18}=gX^{10}+\sum_{l=10}^{17}\left(\prod_{k=l+1}^{17}\lambda_{k,0}\right)
  (\lambda_{l,1}X^0+A^l+M^l).
\]

Thus entry and write-bank displacements add exactly. There is no identifiability assumption or
optimization: the conclusion follows from the executed controlled recurrence. Complexity of the
direct state construction is `O(BTd)` and storage is one entry displacement plus the scalar `g`.
It does not solve the native, unclamped nonlinear suffix because native writes depend on state.

The landed direct v1 result confirms the theorem on the registered prefix at relative error
`3.61e-8--3.69e-8`. Its apparent logit disagreement (`.0967--.1541`) is not a contradiction: the
runner compared every sequence position, including positions after the row's query where dynamic
clamps continued to act but direct replacement was deliberately absent. The correct equality
object is `S_b F`, where `S_b` projects to the registered query's answer/foil logits. Comparing
`F` over a larger intervention footprint changes the mathematical object.

**Executable consequence:** preserve v1 as `arithmetic_or_hook_failure`, preregister a v2
instrument repair with identical rows, interventions, scientific bars, and price, but replace the
all-sequence max-logit diagnostic by max absolute error on the two registered answer/foil logits.
Retain an explicitly reported off-footprint maximum as a diagnostic that cannot gate the route.
The already visible v1 OOD recoveries (`.4582/.4803`) remain below `.5`, so this repair is expected
to establish exact physical equality yet terminate `ood_instability`, not retroactively rescue the
science.

### The margin interaction is an integrated mixed Hessian

For twice continuously differentiable `F`, applying the fundamental theorem of calculus twice
gives the exact identity

\[
I=\int_0^1\!\int_0^1 r^\top H_F(x+sr+tm)m\,ds\,dt.
\]

Janizek, Sturmfels, and Lee develop integrated Hessians as architecture-agnostic pairwise neural
interaction values and show interaction completeness, symmetry, and linearity
([arXiv:2002.04138](https://arxiv.org/abs/2002.04138)). Their attribution path uses a particular
baseline convention; our rectangular two-intervention integral is the direct two-dimensional FTC
specialization, so no attribution or causal-identification claim is imported.

**Object mapping:** `r` is exact residual-identity carriage, `m` is exact complete-module response,
and `F` is the fixed final readout margin. The theorem requires smoothness along the rectangle; the
epsilon-stabilized RMSNorm and tanh satisfy it for observed states. It provides an exact interaction
localization and a curvature integral, not a unique coordinate decomposition of the Hessian. Exact
quadrature would cost backward/Hessian-vector products and is unnecessary while the observed
four-forward finite difference already answers the circuit branch.

**Executable consequence:** after the matched-footprint equality audit, retain the four-cell finite
difference as the canonical readout-interaction statistic. Only if it becomes large and stable
should a Hessian-vector quadrature be opened; otherwise the exact physical two-stream state program
is more informative and cheaper.

### Tensor trains and Hankel realization still solve later, different objects

TT-SVD could compress stored response banks once their operational interface is identified, and a
finite Hankel rank could lower-bound a closed string-to-margin realization once native background
dependence is removed. Neither theorem identifies the present writer, groups its two physical
streams, or supplies selective interventions. Their assumptions remain violated by the open native
attention/MLP background, so both remain post-identification engineering rather than live circuit
discovery.

## Mathematical decision and immediate action

The exact affine-superposition mapping dominates. The result is not a singleton identity circuit:
the selected P7 response prefix and identity state each carry roughly half, compose nearly
additively at the readout, and exchange relative majority under OOD shift. The valid circuit object
is provisionally a two-stream interface, not whichever side barely crosses `.5` on original FIT.

Immediately freeze and implement the matched-footprint v2 audit. If selected logits match while OOD
identity remains below `.5`, preserve `ood_instability` and promote the two-stream state interface:
P7 supplies an interpretable distributed response program and the exact identity term supplies a
direct physical state component. Next test their joint OOD prediction/removal rather than fit a DAS
subspace or compress either stream.
