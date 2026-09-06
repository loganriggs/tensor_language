# Three-hour mathematical review — 2026-09-06 17:26 UTC

## Decision

Treat the circuit as a position-indexed causal program, not a bag of globally patched components. Use projector-to-weight contractions as a cheap structural prior, task-conditioned response tensors as a stronger functional prior, and exact component interventions as the final selector. For DAS, optimize the causal object actually claimed: scalar margin objectives identify quotient directions; full-state claims require centered full-logit/KL fidelity. The next circuit step is convergence plus selective removal, not another subspace rank sweep.

## Position-indexed component programs

For row `i`, component `c`, recipient carrier bank `B_i`, and aligned donor bank `D_i`, define the exact component replacement

\[
z^{(c)}_{i,p} \leftarrow z^{(c)}_{i,p}
  + z^{(c),\mathrm{donor}}_{i,q}
  - z^{(c),\mathrm{base}}_{i,p},
\qquad (p,q)\in B_i\times D_i.
\]

For attention heads, `z` is the selected pre-`c_proj` head slice; for an MLP it is the complete MLP output. Installing several components in one forward makes the program causal-order aware: later native computations see earlier replacements. This is categorically different from the corrected-away query-position intervention, which measured generic final-position influence but could not establish carrier writing.

On heldout/A2 panels, invariant weight scores versus exact carrier effects give attention-head Spearman correlations `0.916/0.699` and top/bottom enrichment `14.7x/9.4x` for has/had and is/was. Thus the head contraction is now a validated discovery statistic.

## Distributed greedy composition

Let `r_i(S)` be normalized donor recovery from the exact jointly installed component set `S`. Greedy selection minimizes

\[
L(S)=\frac1n\sum_i (1-r_i(S))^2,
\]

adding the candidate with the largest strict reduction in `L`. Ten-component paths reach:

\[
\begin{array}{c|cc}
 & \text{heldout A1} & \text{A2}\\\hline
\text{has/had} & 0.7551 & 0.7669\\
\text{is/was} & 0.6944 & 0.8245
\end{array}
\]

with direction fraction one and lower unit-target RMSE than every singleton. Every new is/was prefix from 6 through 10 improves both heldout A1 and A2. The correct next stopping condition is therefore a preregistered marginal improvement `Delta L <= epsilon`, not an arbitrary component-count ceiling. Once the path stops, leave-one-component removal must measure necessity in the joint context; singleton strength cannot substitute for this.

## Exact MLP tensor hierarchy

For a bilinear MLP

\[
y(x)=D[(Lx)\odot(Rx)]
\]

and downstream read map `A`, the complete static contracted tensor is

\[
T_{aij}=\sum_n (AD)_{an}L_{ni}R_{nj}.
\]

Its Frobenius norm is invariant to orthogonal changes of readout coordinates. It exactly replays `A y(x)`, but it measures possible writes, not which gates are active. Empirically, replacing the old `||AD||_F` incidence score with `||T||_F` changes MLP rank correlations only from `0.262/-0.595` to `0.310/-0.476`; static weights remain insufficient.

For paired live inputs `x_b,x_d`, the exact activation-conditioned read response is

\[
\Delta_A(x_b,x_d)=AD\left[(Lx_d)\odot(Rx_d)-(Lx_b)\odot(Rx_b)\right].
\]

Using A1 carrier moments improves development correlations to `0.595/0.310`. On a fully disjoint matched lexicon, A1-conditioned scores predict A2 causal MLP effects at `0.500/0.383`, with MLP4 ranked first by the score for both tasks and first/second causally. This establishes transfer signal but misses the frozen rank/enrichment bars.

## Downstream path sensitivity

The exact first-order behavioral response for component displacement `delta z_c` is

\[
s_c = \left|\sum_{p\in B_i}
  \left\langle \nabla_{z^{(c)}_{i,p}}m_i,\delta z^{(c)}_{i,p}\right\rangle\right|.
\]

This combines the upstream response tensor with the native suffix Jacobian. A central finite-difference check at dose `0.01` agrees within `3.86e-4` relative error for has/had and `3.08e-4` for is/was. Its development rank correlations are `0.95/0.65`, a large improvement over activation conditioning alone. A quarter-dose finite intervention captures more curvature for is/was (`0.667`) but degrades has/had (`0.90`). Therefore higher-order metric machinery is not uniformly superior; exact causal MLP screening is cheap enough to remain the confirmation layer.

This gives a principled hierarchy:

1. static projector/weight incidence for zero-example nomination;
2. activation-conditioned exact tensor response for task typing;
3. suffix-Jacobian contraction for path-aware prioritization;
4. exact full-component patching for final causal ranking.

The first three reduce search cost but do not replace the fourth.

## DAS objective geometry

At one live boundary with orthogonal projector `Q` and exact displacement `d`, the valid partition remains

\[
z_Q=z+Qd,\qquad z_\perp=z+(I-Q)d.
\]

The latest regularization results separate target correction from stability. Tangent noise leaves centered full-vocabulary error nearly unchanged (`0.529 -> 0.519` heldout), so local smoothness is not the main defect. Adding full-vocabulary KL reduces it to `0.0632`, near difference-in-means `0.0592`; on A2 it reduces `0.448 -> 0.245`, near `0.239`. KL also removes the optimized axis's heldout scalar advantage. Thus the optimizer was not failing to search: it was accurately optimizing a quotient objective that omitted most of the claimed state.

For future DAS, define a multi-objective loss such as

\[
L(Q)=\alpha L_{\mathrm{margin}}(Q)
 +\beta\,\mathrm{KL}(p_{\mathrm{exact}}\Vert p_Q)
 +\gamma\,\mathrm{KL}(p_{\mathrm{base}}\Vert p_\perp),
\]

and report the Pareto frontier over `(margin, full-state fidelity, complement inertness)`. A single weighted optimum should not be called uniquely correct unless the frontier and restart identifiability support that claim.

## Next executable mathematics

1. Continue each ten-component carrier program until the best fit improvement is at most a frozen `epsilon`, then seal the converged path.
2. For each selected component `c`, evaluate `r(S\setminus\{c\})`; use the decrement relative to `r(S)` as its conditional necessity, and audit pairwise removals where single removals reveal redundancy.
3. If the converged source program remains far below unit recovery, expand the candidate pool using the validated head weight ranking plus path-conditioned MLP prior, rather than searching all components uniformly.
4. Only after sufficiency and removal stabilize, compress component responses into a predictive tensor program and compare literal stored state, multiply count, and edge count against captured replay and the original model.

Next mathematical review due around **2026-09-06 20:26 UTC**.
