# Three-hour mathematical tensor-network review — 2026-09-08 17:26 UTC

## Exact current object

Bilin18 has residual width `d=1152` and bilinear-MLP hidden width `h=4608`.  At
MLP11, with its deployed normalization included in the input `x_s in R^d`, the
output at scored row/position `s` is

\[
y_s=D\big[(Lx_s)\odot(Rx_s)\big],\qquad
L,R\in\mathbb R^{h\times d},\quad D\in\mathbb R^{d\times h}.
\]

The frozen L7H7/L9H4 writer intervention changes the MLP11 input from `x^0_s` to
`x^1_s`.  Native hidden factor `n` then has the exact output response

\[
o_{sn}=D_{:n}\left[(L_{n:}x^1_s)(R_{n:}x^1_s)
                         -(L_{n:}x^0_s)(R_{n:}x^0_s)\right],
\qquad \Delta y_s=\sum_{n=1}^{h}o_{sn}.
\]

Thus the MLP weight tensor is the architecture-provided CP expansion

\[
T=\sum_{n=1}^{h}D_{:n}\otimes L_{n:}\otimes R_{n:}.
\]

Let `F_c(Y)` be the summed toward-donor answer margin after the unchanged
blocks 12--17, final normalization, tied unembedding, and softcap for construction
`c`.  At the writer-present MLP11 output `Y^1`, define the downstream reader
covector and factor tangent

\[
g_{cs}=\nabla_{y_s}F_c(Y^1),\qquad
t_{csn}=\langle g_{cs},o_{sn}\rangle .
\]

The selected order ranks factors by the worst normalized signed mean contribution
over v17 A2 and v18 A1.  The literal candidate program is the first 16 frozen
indices at one global coefficient `alpha=1.25`; the checkpoint and all downstream
background remain present.  Its allowed confirmation inputs must therefore be a
new capability-gated corpus, not v18 or v19, which selected the dose.

## Contraction graph, degree, gauges, outputs, and price

The tested graph is

\[
(L7H7,L9H4)\to x^{11}\to
\{D_{:n}(L_{n:}x)(R_{n:}x)\}_{n=1}^{4608}\to
Y^{11}\to\text{blocks 12--17}\to\text{answer margin}.
\]

Each MLP11 factor is degree two in its normalized input.  The complete deployed
map is not polynomial: RMS normalization, normalized attention, later bilinear
blocks, and the final tanh softcap remain live.  The tangent is degree one in an
infinitesimal MLP11-output perturbation, not an exact linearization of the whole
finite intervention.

The native representation has reciprocal scaling gauges
`L_n -> a L_n`, `R_n -> b R_n`, `D_:n -> D_:n/(ab)` and hidden-factor permutation.
The rank-one tensor and `o_sn` are invariant to the scaling gauge and equivariant
to permutation.  The earlier residual U8 has an additional `O(8)` basis gauge;
the literal checkpoint basis froze one representative, but that does not make its
coordinates canonical.  Operational causal response under the fixed checkpoint
is the identification criterion.

Outputs to preserve are the signed writer-mediated answer-margin vector, its
direction/cosine, and low answer-preserving P/C movement.  Output reconstruction
is retained as a diagnostic and matched-capacity control, not the target norm.
The incremental executable stores 16 integer indices and one scalar because the
checkpoint supplies the weights.  A standalone literal factor payload would
contain `16(2d+d)=55,296` real weight values plus indices and gain.  Per token it
costs roughly `16(3d+1)=55,312` multiply-accumulate-scale operations, versus about
`h(3d+1)=15.93M` for the complete MLP factor sum; this does not price or replace
the downstream background model.

## Exact theorem and algorithm mappings

### CP uniqueness does not identify these 4,608 factors

Kruskal's theorem gives essential uniqueness of a three-factor CP representation
when `k_D+k_L+k_R >= 2h+2`, up to factor permutation and reciprocal scaling
([Kruskal 1977](https://doi.org/10.1016/0024-3795(77)90069-6)).  Here every
factor matrix has row-space rank and hence Kruskal rank at most `d=1152`, so the
left side is at most `3456`, while the required right side is `9218`.  The theorem's
sufficient condition is impossible.  CP fitting therefore cannot canonically
recover 4,608 semantic pieces from this tensor.  The architecture supplies native
factor labels; causal interchange and the task covector supply operational meaning.

### First-order saliency explains the useful ranking, with a finite-dose remainder

For a factor subset response `v_S=sum_{n in S}o_n`, Taylor's theorem gives

\[
F_c(Y^1+\delta)-F_c(Y^1)
=\langle g_c,\delta\rangle+
\tfrac12\delta^T H_c(Y^1+\theta\delta)\delta .
\]

If the downstream Hessian operator norm is bounded by `M_c` on the segment, the
remainder is at most `M_c ||delta||^2/2`.  Optimal Brain Damage uses a diagonal
second-order weight-loss approximation near a trained optimum
([LeCun, Denker & Solla 1989](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)).
Our score is different: it is first-order saliency of an exact activation response,
with fixed factor coefficients, for a signed causal task effect rather than training
loss at a parameter optimum.  The mapping is exact only to the displayed Taylor
term; the Hessian remainder is the missing assumption.

Optimal Brain Surgeon retains the full Hessian and permits compensating changes
while removing a weight
([Hassibi & Stork 1992](https://proceedings.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)).
Our one-parameter gain is a much smaller compensating family.  Along the top-16
response `v`, a local quadratic model can estimate
`alpha*=-<g,v>/<v,Hv>` for a loss-minimization convention, but no Hessian was fitted
and the registered `1.25` was chosen from observed causal recovery.  It is therefore
a retrospective calibration hypothesis, not an OBS optimum or a proof.

### Output-residual greedy has no applicable submodular guarantee

For exact fixed-coefficient output reconstruction,

\[
J(S)=\left\|\sum_n o_n-\sum_{n\in S}o_n\right\|^2,
\]

and adding factor `n` reduces the residual by
`2<o_n,r_S>-||o_n||^2`.  The signed cross-factor Gram can make this benefit
nonmonotone and violate diminishing returns.  Classical cardinality-constrained
greedy guarantees require a normalized monotone submodular benefit
([Nemhauser, Wolsey & Fisher 1978](https://doi.org/10.1007/BF01588971)).
Those assumptions are not established here, so no `1-1/e` guarantee applies.
Empirically, 256 output-greedy factors still left roughly `.75` training residual
and were nonselective; changing the objective to the downstream task functional,
not tuning greedy depth, was the justified move.

### Worst-construction scoring is a robust heuristic, not unseen-group proof

Taking the minimum normalized signed contribution over two construction groups is
a finite worst-environment score.  Group DRO explicitly optimizes worst predefined
group risk, but overparameterized models can still fail worst-group generalization
without regularization
([Sagawa et al. 2020](https://openreview.net/pdf?id=ryxGuJrFvS)).  Our discrete
ordering is not a group-DRO optimizer and offers no unseen-construction guarantee.
The v19 magnitude miss is exactly the kind of failure that requires prospective
group confirmation.

## Consequence for decomposition of shared read/write subspaces

PCA, SAE, hierarchical SAE, or another activation-only decomposition can still
describe occupancy on a chosen corpus, but its energy objective need not preserve
the computation.  Folding a proven shared subspace into weights is more useful:
it restricts `T` to exact writer factors, while contraction with `g` restricts the
same object on the reading side.  The resulting task-functional tensor answers
which exact weight factors write a causally used direction.  Across tasks, stack
the factor-response matrix `t[task,construction,row,n]` and decompose only after
holding out tasks/constructions; shared components then mean shared causal reader
functionals, not merely shared activation variance.  Any PCA/SAE hierarchy remains
a probe until factor removal, restoration, and cross-task composition pass.

## Executable consequence and opposing predictions

Seal a history-disjoint v20 capability bank before any causal inspection, then run
the exact first 16 stored factor indices at fixed `alpha=1.25` with no refit.  Register:

1. each target construction has recovery in `[.8,1.2]`, positive direction, and
   high cosine, while P/C normalized movement stays below its selectivity bar;
2. finite-dose linearity error
   `|rho(1.25)-1.25 rho(1)| / max(|rho(1.25)|,eps)` is at most `.10` per target;
3. the all-factor and no-op replays retain exact closure.

Passing identifies a prospective fixed-gain partial MLP11 factor program on one
new corpus; it does not yet establish multi-task reuse or standalone sufficiency.
Failure with preserved cosine but wrong recovery falsifies the global gain and
supports a missing construction state.  Failure of cosine/direction falsifies the
shared top-16 task-functional core.  Control movement falsifies selectivity.

This test dominates a new PCA/SAE fit: it directly changes held-out prediction,
within-module splitting, selective manipulation, and stable identification at low
cost.  If it passes, the next mathematical object is a multi-task response tensor
over prospectively chosen circuits, followed by weight-side factor grouping and
joint causal composition.  If it fails, the next object is a tiny construction-
state gain law or a second disjoint factor branch, not an activation-energy sweep.

