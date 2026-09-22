# Three-hour mathematical, organization and efficiency review

22 September 2026, 01:59 UTC. Previous full review: 21 September 22:53 UTC. ACTIVE_TRACK: WEIGHT_FOLDING, under the user's two-day focus through 22 September 15:10 UTC. The previous goal turn only communicated existing evidence; this turn reran a concrete native mathematical falsifier, completed the literature review, and verified the live experiment and queued intervention tests.

## Object and success condition

The model has 18 blocks, residual width 1,152, bilinear width 4,608 and vocabulary size 50,304. The present path selects the bias-free self term of the previous MLP inside both inputs of the last MLP. With zero-based layer numbers, define

$$
m(x)=\lambda_{17,0}D_{16}[(L_{16}x)\odot(R_{16}x)],\qquad
F_v(x)=w_v^\top D_{17}[(L_{17}m(x))\odot(R_{17}m(x))].
$$

Here x is the actual normalized MLP16 input, and the 16 fixed readers w select the earlier output subspace. The implicit coefficient tensor has shape 16 by 1152 by 1152 by 1152 by 1152. It is evaluated with the same x in every input slot. Residual, attention, bias and cross terms are excluded from this selected target; actual normalization and final softcapping remain explicit in native intervention tests. This is not a complete two-block replacement. The prior midpoint program includes a broader set of self/cross terms and remains a separate baseline.

The native path is homogeneous degree four and even in x. Its input-symmetric coefficient tensor specifies its polynomial, but a compact executable representation need not itself be symmetric. Factors admit reciprocal scalings, permutations and internal changes of basis; no recovered channel identity follows from an accurate function alone. RMS normalization approximately fixes squared radius at 1152, with epsilon retained by the actual model.

The objective remains a smaller transparent computation with held-out and OOD prediction, extraction, selective manipulation, stable identification, and composition/reuse. The current 16 coordinates are an operational output basis, not identified semantic units. None of these recent approximations satisfies that entire goal.

## Existing mathematical tools and their actual reach

| Tool | Mapping and assumptions | Cost / guarantee | Limit for this model |
| --- | --- | --- | --- |
| Gaussian active subspaces and conditional expectation | Write x=mu+Sz. Retain eta=V^T z and replace a frozen CP parent by its conditional mean given eta. The Gaussian law and chosen output norm must be specified. | Conditional expectation is the optimal square-integrable predictor for a fixed retained sigma-algebra; gradient spectral tails give an error upper bound. Dense eigendecomposition is cubic in input dimension, after gradient moments are available. | The bound is for the parent under that Gaussian, not the native transformer on text, finite interventions or semantic units. Eigenvectors are not uniquely identified when eigenvalues tie. |
| Fischer / spherical harmonic decomposition | A homogeneous quartic splits into harmonic degrees 4, 2 and 0 multiplied by radius powers. On a fixed sphere, radius factors become constants. | Exact linear decomposition, unique harmonic components. Our cached traces permit the lower-degree part in O(16 d^2) storage/arithmetic per state without expanding d^4 coefficients. | Uniform-sphere orthogonality is not text-distribution orthogonality. Truncation needs an independent accuracy test. |
| HT / tensor train | Factor unfoldings of an explicit multilinear coefficient representative along a chosen tree or chain. | HT storage O(n d r + n r^3) for n comparable-size modes and bounded ranks; SVD-based approximation controls a coefficient norm. | Tree ranks and coefficient gauges can be unfavorable; repeated-input polynomial equivalence, nonlinear normalization and DAG reuse are additional structure. No semantic or minimal-circuit guarantee. |
| Weighted automata / Hankel realization | A multilinear map across an ordered sequence of independent slots can be organized as a linear second-order RNN / tensor train. | Full-rank Hankel blocks allow spectral recovery under finite-realization assumptions; linear algebra is polynomial in the supplied block sizes. Realizations retain state-basis freedom. | A fixed quartic evaluated at repeated x does not supply a proven finite-rank sequence law across all lengths. Minimal state dimension is not minimum arithmetic cost. |
| Arithmetic circuits and equality saturation | Exact distributivity, shared products and homogeneous-component bookkeeping operate on the proposed DAG. | Degree bookkeeping is tractable at degree four; equality saturation represents many valid rewrites. Neither supplies a general efficient globally smallest DAG algorithm. | Approximate covariance corrections and pruning are not exact equalities. Shared-node costs must be charged once; ordinary expression-tree extraction can misprice reuse. |
| Graph-width contraction | Choose a contraction order for an already specified network. | Cost grows exponentially with width (and depends on bond dimensions); useful for contraction scheduling. | It does not discover the smallest network, identify semantic factors or certify causal transfer. Our dense 1152/4608 ports cannot be priced as bounded-dimensional qubit bonds. |

Primary sources: [Zahm et al., vector-valued Gaussian dimension reduction](https://arxiv.org/abs/1801.07922); [Fang and Fawzi, spherical harmonic background in Appendix B](https://www.kunfang.info/pdfs/FF20-MAPR.pdf); [Grasedyck, HT](https://doi.org/10.1137/090764189); [Rabusseau et al., spectral realization](https://proceedings.mlr.press/v89/rabusseau19a.html); [Shpilka and Yehudayoff, arithmetic circuit survey](https://www.cs.tau.ac.il/~shpilka/publications/SY10.pdf); [Willsey et al., egg](https://arxiv.org/abs/2004.03082); [Markov and Shi, contraction width](https://arxiv.org/abs/quant-ph/0511069).

The Gaussian construction already produced a usable program. For four factors l_i+epsilon_i, its conditional mean is the product of l_i, plus six covariance-weighted pair products, plus three covariance-pair constants. Caching pairs needs seven variable products per CP atom. The subsequent lean graph keeps only the two pair corrections whose products are already needed by the quartic, restoring three products per atom. That last deletion is approximate and must be scored against native effects.

## New executable consequence: test the exact spherical lower-degree part

Earlier reviews already covered harmonic metrics and the failure of isotropic spherical source geometry. This is a new native truncation test, not a new discovery of the decomposition theorem.

Let m=E[F(g)] and Q=E[Hessian F(g)]/2 for g standard Gaussian, with one matrix Q per output. Set r^2=x^T x and d=1152. Independently derived and tested formulas are

$$
h_0=\frac{m}{d(d+2)},\qquad
h_2(x)=\frac{x^\top Qx-2r^2m/d}{d+4},\qquad
h_4(x)=F(x)-r^2h_2(x)-r^4h_0.
$$

Both h2 and h4 have zero Laplacian. The proposed cheap approximation retains r^2 h2+r^4 h0; on the normalized sphere it is a quadratic plus a constant. Opposing outcomes: a small remaining h4 would justify developing this direct native quadratic program; a large remainder rejects this truncation without rejecting the exact identity.

The CPU audit was rerun this turn using cached exact native traces. Three independent small native-network controls verify harmonic Laplacians and trace replay, with largest discrepancy 1.37e-12. On position 31 from each of 256 already-opened documents, the approximation has **138.47% value error**, versus **97.84% for the constant part alone**. Its existing root1 same-token response error is **112.55%**. Radius-squared deviations are at most 2.54e-7 relative. This is a strong negative for the proposed truncation. Individual component norms show cancellation on text; sphere orthogonality cannot be imported into this nonuniform sample.

[Executable audit](direct_tensor_match/audit_spherical_harmonic_quartic.py) and [numerical receipt](direct_tensor_match/SPHERICAL_HARMONIC_QUARTIC_V1.json). No new candidate was trained or exported, and no held-out status is claimed for these opened states.

## Decisions, price and outstanding evidence

The lean rank-256 conditional programs use 848,912 floating coefficients including the common writer, versus 2,385,920 for their CP parents. Both use 1,536 variable products. The lean graph reduces dense coefficient multiplications to 828,416, excluding the common writer. Measured CPU speedup is 1.48–2.46 times across tested batch sizes; this is not a GPU or full-transformer benchmark. New-document pooled values are 7.24/7.46%, but these later candidate comparisons use an already-opened panel. Same-token response errors are 10.27/10.55%; selective semantics, genuine domain shift and joint composition remain unestablished.

Highest-information next measurements remain the already-registered full and lean native removal tests, preserving actual normalization, background and softcap. They distinguish an economical local predictor from an economical intervention substitute. The matched uniform-versus-balanced producer optimization tests whether capacity can recover small output coordinates without damaging dominant ones. A lower pooled error alone is insufficient. Do not add another isotropic-sphere fit after the falsifier above.

If removal fails, inspect signed native-versus-candidate effect residuals by amount/domain/output before another fit. If it passes, retain the registered scope and advance to composition and a genuinely untouched shifted-data panel, with features frozen. Neither outcome licenses semantic labels for the 16 basis coordinates.

## Organization and throughput audit

Authoritative recent commits: conditional programs 01:35, lean graph edit 01:42, CPU timing and symmetry redteam 01:47. This represents one constructive program and one graph simplification with subsequent falsification tests, rather than multiple independent discoveries. Earlier shared fitting took 1,205 seconds of GPU execution; parent refitting took 13.94 seconds; new reference capture 1.94 seconds and five-candidate scoring/bootstrap 2.66 seconds. The new harmonic audit takes about one second wall time including startup on this rerun. Queue waiting is separate.

At 01:56, balanced producer PID 1026071 was confirmed live, elapsed about 26 minutes. Full and lean removal hashes d881407e and 9512d6d0 remain in the managed queue behind other authorized work. No live helper was edited and no queue was reordered. Exact thinking/authoring-versus-review minutes were not instrumented, so no invented percentage is reported. This review was bounded to the due checkpoint; the cheap audit reused existing traces rather than adding a GPU campaign.

The decomposition README and explanation index had stale 'current/latest' claims. They are repaired to lead with the latest canonical reports. The computation-path registry and MLP dossier now link this branch; the circuit registry remains unchanged because no new circuit identification occurred. Historical midpoint findings are retained and explicitly separated. No broad refactor is warranted while helpers are frozen for queued jobs.

Handoff to circuit work: use the frozen program's native finite effects to specify which output contribution can be exchanged or removed, then test unrelated behavior and joint reuse. Handoff back to folding: prioritize coefficient structure that predicts those effect residuals, not merely a favorable parent Gaussian score. The full goal remains active.
