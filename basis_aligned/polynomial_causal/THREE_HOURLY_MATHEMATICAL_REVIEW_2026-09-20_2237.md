# Three-hour mathematical review — 2026-09-20 22:37 UTC

ACTIVE_TRACK: WEIGHT_FOLDING (user's two-day weights-first instruction overrides ordinary hourly alternation).

## Current object and executable consequence

We reconstruct a selected pure quartic branch through MLP16, MLP17 and the unembedding. Residual cross terms, biases, attention and native normalization remain outside that branch. The currently extracted four scalar outputs are

$$
p_i=(a_i^\top x)(b_i^\top x),\quad i=1,\ldots,6,\qquad
s_g=p^\top Q_gp+c_g^\top p+b_g,\quad g=1,\ldots,4,
$$

with $x\in\mathbb R^{1152}$ and symmetric $Q_g\in\mathbb R^{6\times6}$. Four shared products $(\ell_j^\top p)(r_j^\top p)$ implement all four quadratic forms. In original inputs this is degree four plus degree two and a constant. Native RMSNorm and softcap are explicitly nonpolynomial. Real unconstrained inputs define the algebraic audit; empirical probes use native normalized states.

Input-product rescaling and exchange, root rescaling, output-basis changes and repeated-input tensor nullspaces complicate uniqueness. The canonical output frame is frozen operationally; it does not establish monosemanticity. The actual shared scalar graph costs 13,916 coefficients and 10 products; fixed residual writers add 4,608 coefficients. Linear additions and edges are also recorded by the DAG cost model. Native background computation is additional.

**Executed restriction that is solvable exactly:** for one real symmetric $Q$, the minimum number of products of linear forms is

$$
k_{\min}=\max(n_+(Q),n_-(Q)).
$$

Each symmetrized outer product contributes at most one positive and one negative eigenvalue. Pairing positive and negative squares via $a^2-b^2=(a+b)(a-b)$ attains the bound. This is a restriction to fixed primitive coordinates and a linear readout, not arbitrary arithmetic-circuit optimality. Numerical eigendecomposition costs $O(6^3)$; exact rational congruence also uses cubic arithmetic operations, with coefficient bit growth. The representation is not unique.

The first checker incorrectly required eigenvalues to be separated from zero. Exact rational arithmetic on the stored FP32 coefficients shows rank six for all four forms, with inertias $(4,2),(3,3),(3,3),(2,4)$. Tiny eigenvalues make numerical rank different from exact rank. The revised checker verifies the exact inertia, and labels tolerance-based exports as approximations. Dropping eigenvalues below $10^{-10}$ of the largest magnitude gives standalone total product counts 9,8,8,9; FP32 export replay errors stay below $9\times10^{-8}$ on reused actual panels. These errors compare against the extracted program, not the native model.

**Joint reuse is the decisive result.** The four $Q_g$ span a four-dimensional matrix space, verified over exact dyadic rationals. Any $k$ shared quartic functions with linear output readout span at most $k$ dimensions. Thus at least four roots are necessary, and the existing shared four-root implementation attains that bound in this class. An exact nonzero 12-by-12 minor of the input readers proves the six primitive products can vary independently for unrestricted $x$. Independently simplified scalar roots would use 16 products after sharing the six primitives, versus the existing 10.

The successor CPU analysis was executed: a three-root approximation has at least 24.2% relative coefficient Frobenius error in the fixed primitive/output coordinates. After allowing a free constant and quadratic readout on reused panels, the finite-sample singular-value floor is 20.7–21.0% of residual quartic variation, but only 3.2–3.3% of total scalar norm. This denominator distinction matters. These necessary bounds need not be attainable with three product roots and do not bound native intervention error or population risk.

## Literature search and mappings

Search covered tensor/HT decompositions, bilinear and arithmetic-circuit complexity, Hankel minimal realization and graph-width contraction. No claimed general solution to circuit discovery follows from these connections.

| Candidate | Mapping and guarantee | Cost and violated/missing assumptions |
|---|---|---|
| Bilinear complexity | Root functions correspond to symmetric products of linear forms. Output-span rank is a necessary lower bound, attained here by four existing shared roots. | Our rational 4-by-36 rank test is small; arbitrary tensor-rank search is a different, harder problem. Independent input slots in standard bilinear algorithms differ from repeated $p$. No uniqueness or semantic guarantee. |
| HT / hierarchical SVD | The two-level quartic program is comparable to a hierarchy grouping four tensor slots. Prescribed-tree ranks organize a compact coefficient representation. | Dense order-five expansion at width1152 is infeasible; small-core calculations are viable. Standard coefficient approximation treats slots independently and does not optimize reuse between branches or polynomial equivalence on repeated inputs. |
| Weighted automata / Hankel realization | A finite-rank Hankel object can identify a minimal linear state representation for a weighted language, subject to its realization assumptions. | We have a fixed commutative polynomial and nonlinear native background, not a demonstrated finite-rank rational series over concatenated words. No Hankel construction or state-recovery guarantee is established here; not adopted as an algorithm for this object. |
| Contraction graph width | Given a tensor network, contraction ordering can control intermediate storage. | Optimizing evaluation of a supplied graph does not discover a smaller polynomial DAG or semantic features. Its benefit would be scalable implicit losses for deeper targets, not a replacement for the current four-output search. |

Primary sources: [Ye and Lim, structured matrix computations and tensor rank](https://arxiv.org/abs/1601.00292); [Grasedyck, hierarchical SVD](https://epubs.siam.org/doi/10.1137/090764189), also listed on the [author's publication page](https://www.igpm.rwth-aachen.de/team/grasedyck/publications/ta); [Balle, Panangaden and Precup, canonical weighted automata](https://arxiv.org/abs/1501.06841); [Markov and Shi, tensor-network contraction](https://arxiv.org/abs/quant-ph/0511069). The inertia and output-span arguments above are derived directly for our object; they are not attributed as general discovery theorems from these papers.

## Direction and circuit implications

Retain the shared four-root program. Standalone exports are useful extraction alternatives, but further exact root-count search with fixed primitives is exhausted within the proved class. Approximation or a changed dictionary remains possible. The more informative remaining question is which input computations and metric recover code feature1 without sacrificing other features. Same-token swaps support contextual effects, but task selectivity, semantic identity, robust OOD transfer and full composition remain unproven. Lower reconstruction error alone cannot close those requirements.

Circuit handoff: retain the fixed scalar interface and native recipient background for future targeted swaps; do not infer a semantic label from output sharing. Folding handoff: change the primitive dictionary or optimize a feature-sensitive objective rather than chase an impossible exact three-root realization.

## Organization and efficiency

Primary receipts live once under `direct_tensor_match`: `SCALAR_ROOT_INERTIA_V1.json`, `SCALAR_INERTIA_PROGRAMS_V1.pt`, and `SHARED_ROOT_RANK3_FLOOR_V1.json`, with reproducible CPU scripts. The timed 22:33 report correctly records the checker failure before this repair; preserve that chronology and link this review as its resolution. Do not relabel the reused confirmation panels as fresh for subsequent swaps.

This review's executable analyses took roughly one second per CPU invocation; most elapsed time was reasoning, implementation and source review. No GPU job or duplicate native evaluator was added. The corrective implementation included known zero, indefinite and singular inertia controls; no large new test framework. Previous reports already identified authoring overhead; combine this result and its lower-bound consequence into one durable unit. Global registries were not comprehensively audited in this bounded review; the local report/receipt links are maintained, and no global consistency claim is made.
