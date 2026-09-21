# Three-hour review: algebraic proposals, approximate reuse, and identification

21 September 2026, review started 13:50 UTC and recorded at the next safe boundary. Full goal remains predictive, OOD-valid, extractable, selectively manipulable, reusable, simple and stably identified circuits. The user-authorized two-day weights-first focus remains active; no early completion or track switch is inferred.

The previous goal turn made substantive progress. Since the 10:50 review, restricted graph fits, private-span bounds, coordinate diagnostics and two-stage algebraic recovery have changed the next actions. The broad folded contribution and full-model goal are still unsolved.

**Exact object, geometry and cost**

Six symmetric matrices Q_o of size1152x1152 define source reads q_o(z)=z^T Q_o z. Each pair feeds one of three selected components:

$$
\phi_j(z,h)=\left(\frac{h^Ta_j-q_{2j}(z)/2}{s(h)}-\alpha_j\right)
\left(\frac{q_{2j+1}(z)}{s(h)}-\beta_j\right).
$$

The supplied native states z and h remain separate inputs. RMS scaling s(h), later normalization and softcap remain explicit. Source replacements retain their mean/affine corrections. Before explicit denominators, this includes polynomial terms through degree four; the source-only tensor is order three (six outputs, two1152input modes). It is not the complete network polynomial.

The graph has three64-dimensional input dictionaries reused by pairs of consumers, plus224private directions per pair. Each pair's shared128-dimensional core and private224-dimensional core are compiled into scalar products. Actual price:1056products,1058124floating coefficients,1047648source multiplications. Original baseline1330560source multiplications; the20%-saving ceiling is1064448. These counts exclude common interface work on both sides and do not imply a measured whole-model speedup.

Input bases admit invertible gauges, and quadratic coefficients have tied symmetric input slots. Native and covariance-shaped errors use different input geometries with pairwise normalization; covariance comes from calibration. Neither equals all normalized component errors or the full lifted moment metric automatically.

**Literature mapping and limitations**

1. [Fang et al., simultaneous block diagonalization by congruence](https://arxiv.org/html/2503.01166v1): local pairs of symmetric forms map to congruence blocks. Our regular-pair compiler gives candidate invariant input spaces. An individual pair with a nonsingular pencil base reduces to generalized eigenproblems, with polynomial dense linear-algebra cost. Grouping candidate blocks across consumers adds a separate combinatorial problem. The general center theorem does not certify uniqueness of our overlapping dictionaries, nor does the previous all-six-form center obstruction exclude pairwise reuse.

2. [Cai and Li, identification of matrix joint block diagonalization](https://proceedings.mlr.press/v130/cai21a.html): a common full-column-rank mixing matrix and block-diagonal cores map to each local pair after input support restriction. The union of every shared/private dictionary can be overcomplete, so one global independent mixing matrix is not available in the toys. Exact and noise-related identification assumptions cannot be imported wholesale. Their result does not assert that an arbitrary beam search recovers our graph.

3. [Spielman, Wang and Wright, sparsely used dictionaries](https://arxiv.org/abs/1206.5882): in a pair-pencil eigenbasis, discovering a common invariant group can be viewed as sparse block-coordinate support discovery. Their polynomial recovery result uses an invertible square dictionary and a random sparse coefficient model. Our signed, deterministic trained coefficients and potentially overcomplete dictionary do not satisfy those assumptions. Sparse vectors in a subspace are a plausible proposal mechanism, not an applicable guarantee.

4. [Reconstruction algorithms for low-rank tensors and depth-three multilinear circuits](https://arxiv.org/abs/2105.01751) connects tensor rank to sums of products of linear forms, giving efficient reconstruction for specified constant-rank classes. This maps to a restricted arithmetic stage, not an arbitrary reused DAG with hundreds of products, repeated inputs and normalization. Constant-small-rank assumptions do not hold for the current1056-product candidate. General arithmetic reconstruction remains broader than coefficient compression.

HT/Tucker organizes tensor slots and subspaces; it does not select reuse across consumers or remove repeated-input polynomial gauges. Weighted-automaton/Hankel realization would require concatenation-indexed observables that these six static forms do not supply. Those alternatives remain useful elsewhere, but do not replace the concrete pair algebra here. The user's tensor-similarity objective provides a fitting metric, not an identifiability theorem.

**Executed mathematical consequence**

For a local pair T_o=F diag(A_o,C_o)F^T on its support, the operator formed by one pencil combination times the inverse of another is similar through F to a block-diagonal operator. Thus common/private input spaces can be sought among invariant groups of pencil blocks. Exact support intersections across consumers reject incompatible groups.

This target-only procedure recovered all five planted shared spaces to numerical precision. Fifteen coordinate/output-mixing controls passed actual compiled execution and shared-projection cost savings. Exhaustive enumeration costs O(2^b) in pencil-block count b and is unsuitable at native widths. Exact filtering also failed its noise amplification criterion (3/5 versus4/5 required at1e-8noise).

The approximate version truncates pair supports at architecture width, scores compatibility, and locally refits the graph. Exhaustive and beam16 variants pass coefficient gates on5/5 at0,1e-4 and.01noise, and subspace identification on4/5 at.01noise. Beam4 fails one zero-noise case and identifies only3/5 noisy subspaces. Its low function error with a wrong shared space is direct evidence that fidelity alone does not identify internal features. Bounded dynamic programming retains O(beam*width) partial candidates; heuristic pruning removes any optimality guarantee.

Core-energy regularization separately prevented all60 singular fits. Stronger penalty recovered more cases, but the registered geometric-mean selector chose an easy-case-dominated setting that failed recovery. No retrospective promotion. Exact oracle controls and deterministic failure replay separated bad optimization paths from unrepresentable targets.

**Native result and next decision**

The just-completed four-arm controlled refit (54.39seconds) compares inherited and algebraic proposals at the same L-BFGS budget. All instruments and cost checks pass; component and coefficient fidelity fail. Covariance errors: inherited7.8931%, algebraic8.1935%. Original-coordinate fit errors: inherited49.7619%, algebraic50.0072%. The algebraic-initialization advantage prediction fails. Primary component errors2.218%,2.294%,10.672% remain above original limits. Native proposal quality, not toy recovery, governs adoption.

Next bounded native question: does forcing equal private widths waste the remaining legal arithmetic budget? The current graph has16800source-multiplication headroom before the original20%-saving ceiling. Each additional private direction costs1155source multiplications (1152projection+1product+2readout), so at most14directions fit. This is a fixed-total-cost allocation hypothesis, not permission for an unpriced rank sweep or a relaxed fidelity bar. A CPU residual-direction/profile test can allocate this capacity from weights, preserve all original component/derivative/dual-metric guards, and reject the hypothesis cheaply if it does not help. If a faithful candidate emerges, freeze it before fresh/OOD/selective/composition tests and compare with an appropriate matched-cost pair baseline.

**Organization and efficiency audit**

The canonical overview remains the01:04filename; the13:45report links the new algebraic route, failures and raw receipts. Exact discovery, noisy recovery, selection-rule misses and native initialization/refit have distinct artifacts. No failed instrument has been overwritten: V1 non-contiguous L-BFGS failure, V3 skipped zero-noise refit assumption, singular support runs, and beam4 misses remain explicit. The source helpers reuse pair compilation, graph execution, pricing and assessment rather than rebuilding those interfaces. The new method is a computation-path candidate, not a semantic circuit dossier.

Measured work includes174.98seconds for60regularized toy fits, about0.03seconds for exact grouping,4.38seconds for the corrected exhaustive noisy suite, and54.39seconds for the controlled native refit. Earlier private-coordinate native fitting took280.60seconds. Exclusive authoring/review wall-time fractions are not available; do not invent them. A real inefficiency was duplicated tiny fit runners and inconsistent failure handling. The support runner now preserves partial valid and invalid arms; further framework work is deferred in favor of the bounded scientific comparison. The GPU job is terminal; the next CPU capacity analysis is claimed immediately.

Circuit handoff: neither shared subspace fidelity nor low coefficient error establishes semantic units. Retain scalar constituent conditions, require stable operational equivalence, and test edits selectively after native fidelity. Weight-folding handoff: inspect the14-direction cost allocation before more generic optimizer retries. The full goal remains active.
