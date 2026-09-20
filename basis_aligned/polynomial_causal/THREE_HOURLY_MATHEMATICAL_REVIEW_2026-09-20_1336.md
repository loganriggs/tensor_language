# Mathematical review: source–reader reduction and prospective failure

Goal remains a simpler executable circuit with OOD prediction, independent extraction, selective manipulation, composition/reuse and stable identification. No complete circuit is established.

## Current mathematical object

The model has 18 blocks, residual width1152, bilinear width4608, vocabulary50304. Native bilinear writes are D[(Lx)*(Rx)]; attention multiplies two QK products with V (degree5 before normalization). RMS, Q/K normalization, cache, residual coefficients, biases and final30tanh remain explicit. Two pure bilinear layers give a quartic output tensor H[v,i,j,k,l]. The full normalized model is not a fixed polynomial.

At the present pre11 source interface, each input supplies D_c in R^(23x1152) and nine canonical output readers R_c in R^(9x1152), with G_c=R_c D_c^T. The 23 ports refine six residual sources while preserving their rounding gauge. Inputs are nominated subject/attractor noun positions; background states, x0 and first-layer V remain native. Eight control contrasts accompany number. A shared state projector Pi could expose intermediate features through R_c Pi D_c^T, retaining context-dependent readers instead of averaging them. This is a conditional tangent object: producing D and R still costs native computation.

## LITERATURE_SEARCH

Actual searches: “Rowley 2005 balanced proper orthogonal decomposition model reduction snapshots adjoint”; “Benner Goyal Gugercin quadratic bilinear systems H2 model reduction arxiv”; “site.princeton.edu Rowley 2005 balanced proper orthogonal decomposition pdf”. Opened primary sources:

- [Rowley 2005, institutional record](https://collaborate.princeton.edu/en/publications/model-reduction-for-fluids-using-balanced-proper-orthogonal-decom/): snapshot balancing is a computational baseline. This review did not inspect its full proof.
- [Otto, Padovan, Rowley, CoBRAS, full text v3](https://arxiv.org/html/2207.14387v3): theorem3 gives a factorized optimal projection for a covariance-weighted residual. It requires nonzero retained singular values; no unique semantic basis follows. Theorem2's nonlinear conditional-expectation error bound assumes Gaussian inputs with positive definite covariance and regularity. Our finite intervention snapshots do not establish those distribution assumptions, and native evaluation at a projected state is not that conditional expectation. We use the algebraic theorem, not a nonlinear fidelity guarantee.
- [Benner/Goyal, quadratic-bilinear balanced truncation](https://arxiv.org/abs/1705.00160) and [Benner/Goyal/Gugercin, H2 reduction](https://arxiv.org/abs/1610.03279): read abstracts. Continuous-time QB control-system results are not directly applicable to our finite-depth, normalized, context-dependent transformer; no stability or H2 bound is imported.
- [Grasedyck, hierarchical SVD](https://www.mis.mpg.de/publications/preprint-repository/article/2009/issue-27): revisited primary abstract as the tensor baseline. Small unfolding ranks support compression, not identification of sparse reusable computations.

Exact finite-snapshot mapping and executed consequence: concatenate opposite-panel source columns into X[1152,1104] and reader columns into Y[1152,432]. If Y^T X=U Sigma V^T, set Phi=X V_r Sigma_r^(-1/2), Psi=Y U_r Sigma_r^(-1/2), Pi=Phi Psi^T. Then Psi^T Phi=I and Y^T Pi X is the truncated SVD. This minimizes ||Y^T(I-Pi)X||_F over rank-r projectors. Dense formation costs O(d mn), SVD O(mn min(m,n)); deployment stores2dr scalars and applies two dense projections, versus dr for orthogonal POD. Singular subspaces have rotation/sign gauges and may be unstable near repeated singular values; oblique projection can amplify unseen directions. The objective mixes all source/reader snapshots, not only paired native contexts. Held paired-context errors must therefore be measured separately.

## BASELINE_COMPARISON

Executed audit_balanced_source_readers_v1.py on saved FP64 tensors; fit only opposite, evaluate already-opened congruent. Worst per-output relative Frobenius gradient error across role/template groups:

| Rank | Balanced | POD same rank | POD same storage (twice rank) |
|---|---:|---:|---:|
|8|1.0232|1.1004|0.9509|
|16|0.8995|0.9509|0.7331|
|32|0.6523|0.7331|0.7606|

All remain poor; this is an exploratory baseline, not a preregistered native adoption test. No cherry-picked output or per-input fitting. Saved BALANCED_SOURCE_READERS_V1.json records all nine outputs, prices and replay errors. Prior v653 already tried snapshot balancing in a different suffix response compiler and failed width8; this new receipt is limited to the 23-port source interface, not novelty of the algorithm.

Existing same-object tensor baseline remains essential: native small-interface quartic canonical exact storage280; rank8 HT656 at9.35% coefficient error; rank2 HT116 at36% error. HT has not beaten exact canonical storage there. These coefficient errors are not comparable numerically to today's gradient errors. Full native generators remain charged. See DECOMPOSITION_BASELINES_2026-09-20.md.

HT represents a tensor as a tree of smaller bilinear computations: linear leaves combine into quadratic features, then quartic features and beyond. The tree groups tensor slots, not disjoint coordinate subsets; each leaf may see all x. Our desired extension uses sparse bilinear cores, adaptive widths and shared features in a DAG. Core sparsity counts interactions; leaf sparsity counts coordinate usage. Width, nonzeros, shared computations and native generators must all be priced. For repeated inputs only Sym(H) identifies the polynomial, but forcing a symmetric representative can inflate tree ranks: (x^T x)^2 has the compact representative delta_ij delta_kl. Thus polynomial matching may use ||Sym(H-Hhat)||_F while Hhat stays unsymmetrized. No rank or sparsity objective alone establishes reusable causal units.

## REDTEAM_POSITIVE

Fresh v3 frozen role-bank test: selectivity10/16 cells (subject8/8, attractor2/8), prediction0/16. Subject worst collateral ratio0.08751 and min retention1.092 are positive at the declared interface, but do not establish correct grammatical computation. Opposite along_with singular has native capability3/6; all other distinct cells6/6. Both roles repeat that same capability measurement; do not count it twice as independent evidence. No failed rows removed. Counterfactual source generation still requires native prefix calls. No prediction, composition, independent full extraction or stability promotion.

## REDTEAM_NEGATIVE

Frozen v3 test replays old v2 effects exactly; refined-port collapse is exact and call counts match18prefix/30native. Failure is not explained by that replay or a silent held-data refit. Baseline native weakness remains a semantic confound, not an excuse to drop cells. Balanced snapshot audit checks biorthogonality and exact truncated cross-snapshot contraction; a planted low-energy/high-observability coordinate is recovered. These checks support implementation correctness, but do not rule out every bug or show all balanced ranks fail. No native projection test or full-rank recovery was run here. Results do not prove intrinsic impossibility or lack of representational capacity.

## organization and efficiency

Canonical fresh receipt: source_ood_v3_role_bank_v1_result.json plus runlog and existing preregistration/binding. This review links it into registries and explains capability failures. CPU consequence reuses saved tensors, consumes no GPU and finishes in1.1s; native v3 run reports1.64s. Large authoring overhead relative to runtime argues against another bespoke runner. Reuse shared context builder and observables for later interventions. No broad refactor is justified now. Scheduled cron polls every5minutes and launches only when the substantive three-hour clock is due; exact execution depends on instance availability. No offline reviews are fabricated.

CIRCUIT handoff: retain fresh subject selectivity as a scoped positive, and attractor/prediction failures as blockers. WEIGHT_FOLDING handoff: test contextual reader computation, not just another static mean or rank sweep. Next folding object is the explicit pre11-to-output reader contraction with native normalization denominators retained; candidate intermediate computations need causal transfer, not merely lower gradient error. Snapshot reduction supplies matched-cost controls, not the solution. Hourly alternation remains separately governed. Next mathematical review is due three hours after this receipt.
