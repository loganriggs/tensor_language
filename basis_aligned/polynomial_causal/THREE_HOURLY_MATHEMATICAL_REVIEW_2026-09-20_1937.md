# Three-hour mathematical and literature review — 20 September 2026, 19:37 UTC

Interval:16:37–19:37. User-directed two-day weights-first science remains primary. Goal remains predictive, extractable, selectively manipulable, reusable, stable and simple circuits. None of the current decompositions establishes all these properties.

## Decision and negative-result correction

Do not conclude that the new eight-product centered program dominates the26-product quartic. At48,384 coefficients it improves total panel2error25.28%→23.12%, but worsens centered variation27.81%→34.99%. Its residual mean-energy fraction falls from.030891 to.001183 while centered residual energy rises.033025→.052284. Thus its aggregate improvement is a mean improvement at the expense of variable response.

Next: use exact noncentral Gaussian moments from the weights to add a constant to a frozen quartic student, retaining its stronger variation. An output-rank8 student needs46,160 coefficients; adding1,152 bias coefficients totals47,312, below the existing48,384 budget. Its26products remain explicit. Compare Gaussian-derived correction with an empirical calibration-mean diagnostic and an evaluation-panel oracle bound; only the first is the primary weights-first candidate. Do not train on or select using evaluation-panel means.

## Mathematical object and metric

Native selected path:

$$F(x)=C\{[LD((Ax)\odot(Bx))]\odot[RD((Ax)\odot(Bx))]\}.$$

Input and orthonormally reduced output dimensions are1152; native bilinear widths4608. C includes unembedding's reduced frame and final down projection; D includes the previous down projection and carry multiplier. This is the pure degree-four term, not the complete two-block model. RMSNorm, attention, other residual paths and softcap remain explicit boundaries. Its order-five coefficient tensor has four repeated input slots. Native factors give a compact unsymmetrized representative; only its input-symmetrized part defines the polynomial.

The current centered candidate is `c + Wlin Rlin (x-mu) + Cq[(Aq(x-mu))*(Bq(x-mu))]`, with linear rank8 and quadratic width8. It stores48,384 FP32 scalars including mu and c, excluding the common native output frame. Its fitting metric is the centered Gaussian Taylor-degree<=2 function loss after optimizing the constant. It is not the original global coefficient loss or the full empirical prediction loss.

For symmetric quadratic error tensor Δ, covariance M, and linear error J, the free-constant Gaussian error is

$$2\|\Delta\times_1 L\times_2 L\|_F^2+\|JL\|_F^2,\qquad M=LL^\top.$$

Our independent quadrature control gives zero discrepancy at reported FP64 precision. This justifies cross-allocation selection by linear gain plus twice quadratic coefficient gain. It does not justify comparing different M values using raw gain.

The arithmetic DAG has learned linear nodes, product nodes, one constant, shared references, and degree bound<=4. Cost counts reachable products, additions and coefficients once, with structural unit signs declared separately. Feature scaling, factor exchange and changes of intermediate basis remain gauges. No monosemanticity or unique factor identification follows from these costs.

## Primary-literature matches and limits

**Separable least squares / variable projection.** Our model is Y≈Φ(theta)C, with nonlinear internal features and linear output coefficients. Eliminating C is directly the separable least-squares construction discussed by [Golub and Pereyra, author-posted paper](https://www.researchgate.net/publication/216212172_Separable_Nonlinear_Least_Squares_the_Variable_Projection_Method_and_its_Applications) and [O'Leary and Rust, author manuscript](https://www.cs.umd.edu/~oleary/software/varpro.pdf). For n probes,k features,V outputs, forming a dense normal system costs O(nk²+nkV), followed by O(k³+k²V) work. These are our accounting estimates, not a new theorem. A fixed full-rank feature matrix gives a unique writer; rank changes and ill-conditioning require care. Eliminating writers does not make internal feature optimization globally convex or identifiable. Our paired square control yields8/8 vs5/8 recoveries below1%, conditional on its grid. Native coefficient-space solves use analogous exact self/cross products and an explicit ridge.

**Exact Gaussian moments.** [Isserlis's original paper](https://academic.oup.com/biomet/article-abstract/12/1-2/134/193428) supplies the Gaussian pairing rule; shift by mu for noncentral moments. For quadratic h_i=xᵀQ_i x,

$$m_i=\mu^\top Q_i\mu+\operatorname{tr}(Q_iM),\qquad
\operatorname{Cov}(h_i,h_j)=2\operatorname{tr}(Q_iMQ_jM)+4\mu^\top Q_iMQ_j\mu.$$

Then E[h_i h_j]=m_i m_j+Cov(h_i,h_j), so a bilinear readout's mean is computable without expanding the quartic. For native first-layer channel products, pairwise covariance requires O(k²d+kd²) arithmetic and O(k²+kd+d²) intermediate storage, before contraction through native D,L,R,C. This solves the **Gaussian mean only**. Normalized text inputs are neither Gaussian nor independent draws; covariance alone does not determine their fourth moments. The new executable helper passes independent quadrature at2.0e-16(native) and6.4e-16(bank).

**Equality saturation and shared-cost extraction.** [egg](https://arxiv.org/abs/2004.03082) represents many exactly equivalent expressions and supports domain analyses. It maps to our distributive/caching rewrite proposals, not approximate feature merges. [Goharshady, Lam and Parreaux](https://doi.org/10.1145/3689801) give optimal extraction algorithms parameterized by low e-graph treewidth and establish hardness of unrestricted extraction. Our desired cost is DAG cost: shared nodes paid once. A tree-additive extractor can misprice it. The theorem applies to a finite represented equivalence graph, not all possible learned real coefficients or arithmetic programs. We have not built such an e-graph or verified a low-treewidth assumption. For now, global reachable-cost evaluation over bounded candidates is cheaper than importing a full equality engine; no global optimum claimed.

**Tensor-network contraction width.** [Markov and Shi](https://arxiv.org/abs/quant-ph/0511069) link efficient simulation to bounded graph width. The relevant analogy is the tensor network for a **loss contraction**, including teacher/student copies and metric connections—not merely the cheap forward arithmetic DAG. Low forward product count does not certify cheap exact loss evaluation. Their bounded-dimension circuit assumptions do not directly supply a polynomial-time algorithm for our arbitrary learned graph. An actual contraction graph and dimension-weighted intermediate cost would be required before applying a width guarantee. Synthetic probes remain a valid alternative functional objective.

These matches offer algorithms for restricted subproblems, not a theorem that semantic circuits must emerge. Weighted automata/minimal realization or identifiability analogies do not currently improve on the explicit moment and variable-projection controls: our inputs are continuous repeated polynomial slots and the model includes nonlinear normalization outside the selected path.

## Executable consequence and opposing predictions

For a fixed student g and reference input law, the unique optimal additive correction is b=E[F−g]. It changes only the mean, leaving centered predictions invariant. Implement exact Gaussian teacher/student means, verify on small quadrature, then evaluate a frozen native quartic plus b. If Gaussian correction fails while the empirical calibration correction succeeds, higher-moment/model-law mismatch is implicated. If both fail on panel2, mean transfer is unstable. If the corrected quartic dominates the centered model at similar cost, the apparent advantage of degree truncation was largely unpriced constant flexibility.

The helper `direct_tensor_match/gaussian_quartic_mean.py` is implemented and independently checked; the native comparison is the next managed run. No rank or hyperparameter will be selected from that evaluation.

## Organization and efficiency

Authoritative receipts since16:37 include original-coordinate weighted fits, exact input-span ceilings, centered degree census, native mixed-root replay, graph-sharing failures, and now centered compact fits. The latest timed report is the19:23 file with addenda; raw receipts remain in direct_tensor_match. The computation-path registry and research index link these. The implementation specification has a dated update distinguishing implemented operations from remaining general graph search.

Recent timings: fixed-graph10fits~8s; native centered26arms~20s; deployment graph replay~1s. Native runner started19:34:00 and exited19:34:21. Most elapsed time is analysis/authoring, not GPU computation. Reusing captured rows and teacher targets avoided repeated model forwards. No new bespoke harness is warranted; reuse the moment helper, existing exact writer solves, and one-round edit/refit loop. The residual audit is essential because it reversed the initial interpretation of a favorable aggregate result.

Circuit handoff: compare centered variation and response to feature removal for frozen candidates before assigning meanings. Folding handoff: preserve exact scope and evaluate mean-corrected quartic versus centered quadratic at literal price. Full causal/OOD validation remains pending under the user's exploration focus.
