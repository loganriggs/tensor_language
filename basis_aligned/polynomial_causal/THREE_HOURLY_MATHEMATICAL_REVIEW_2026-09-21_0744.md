# Mathematical review: stable functions versus stable product dictionaries

21 September 2026,07:44 UTC. Previous three-hour review04:35. Primary-literature search and executable consequences completed; full circuit goal remains active.

## Current object

Four selected quadratic reads form T in R^(4 x1152 x1152), symmetric in its input slots. They feed two native component scalars; a third component retains a private pair. The fitted shared tensor is

$$
\widehat T=\sum_{r=1}^{256}w_r\otimes\operatorname{sym}(l_r r_r^\top).
$$

Input covariance whitening and per-output-pair scaling define the coefficient inner product. Exact centered affine and mean corrections remain separate. Source reads are quadratic; their outer products have degree up to four before explicit RMS division. The program still needs native earlier z and later h. The shared graph plus private branch uses512source products and897804floating coefficients; the baseline768products has the same floating storage. This does not price upstream input production as free for whole-model claims.

Gauges include atom permutation, left/right swap, reciprocal factor/readout scales and more general function-preserving decompositions. Input symmetrization identifies the polynomial, but does not require the stored computational representation to be symmetric. A general input-basis change may destroy cheap product structure.

## Variable projection: an exact conditional solve, not global recovery

Our readout is linear at fixed product directions. Solving its ridge least-squares system eliminates these linear variables, leaving a nonlinear optimization over directions. This is the variable-projection setting described by [Golub and Pereyra](https://www.researchgate.net/publication/216212172_Separable_Nonlinear_Least_Squares_the_Variable_Projection_Method_and_its_Applications), an author-uploaded primary review. The correspondence is exact for the chosen coefficient objective. Our envelope gradient includes the ridge penalty and has five independent gradient checks. It does not guarantee global convergence or identifiable factors.

At n=256products, Gram construction costs O(d n^2), its solve O(n^3); contractions against dense target matrices add O(o d^2 n). Positive ridge makes the readout solution unique for fixed directions. That uniqueness does not extend to the learned directions. Toy restart failures and native metric mismatch remain possible.

## CP uniqueness and ill-posedness do not settle this case

[Kruskal's primary paper](https://www.sciencedirect.com/science/article/pii/0024379577900696) studies uniqueness of trilinear decompositions. Each symmetric mixed product here expands into two CP terms with duplicated output direction. Consequently, standard sufficient Kruskal-rank conditions cannot directly certify this expanded representation. Failure of a sufficient condition does not prove nonuniqueness. We also approximate a higher-complexity target, rather than exactly decompose a known low-rank tensor.

[De Silva and Lim](https://arxiv.org/abs/math/0607647) show that best low-CP-rank approximation can be ill-posed. This is a relevant warning, not an explanation established for our fits. Current product Gram conditions23.7–26.1, bounded readouts and FP32/FP64 discrepancies below3.3e-6 provide no evidence of catastrophic diverging-component cancellation. Function and atom stability require direct tests.

## Principal angles give an executable distinction

[Björck and Golub](https://rainbow.ldeo.columbia.edu/~alexeyk/BjoerckGolub1973.pdf) compare subspaces using singular values of cross-products of orthonormal bases. Map each column to a vectorized joint quadratic atom. We need not materialize those huge vectors. If A and B contain them implicitly, their Gram matrices follow from

$$
\langle w\otimes\operatorname{sym}(lr^\top),v\otimes\operatorname{sym}(ab^\top)\rangle
=\frac{w^\top v}{2}[(l^\top a)(r^\top b)+(l^\top b)(r^\top a)].
$$

Let X,Y whiten the positive Gram eigenspaces. Singular values of X^T(A^TB)Y are principal cosines. This is our Gram-based implementation of the subspace comparison; eigenvalue truncation and Gram conditioning must be checked. Cost O(d n^2+n^3), memory O(d n+n^2). It tests equality of linear spans, not equality of cheap arithmetic programs or semantic units.

Executed `audit_profiled_subspaces.py`: invertible-mixing control passes; rotated-coordinate null has no cosines>=.90. Across the actual fits, medians are.121–.132 and only3–4/256directions exceed.90. Thus instability is not merely a change of basis within an otherwise fixed full dictionary. Nevertheless each total fitted tensor projects into the other span with4.18–4.43%relative error.

Executable extension: average the alternate orthogonal projectors restricted to the reference span. Retain eigenvectors with mean projection energy>=.81 and independently check each alternate. Three retained directions carry99.9937% of the reference total-function energy. Worst restricted-subspace retention is.8105. This passes the registered descriptive consensus screen, but it is partly expected because fits share a target. It neither yields three scalar products nor proves feature semantics. Re-executing that coefficient consensus with original affine corrections is the next actual CPU step.

## Metric and broader representation audit

The [user's paper, Appendix A.3](https://arxiv.org/html/2605.15183v1) uses input moments to define functional geometry. Our covariance-weighted coefficient norm is not a general substitute for those moments. Independent Gaussian trace audit finds the missing mean correction contributes only about0.12% of profiled-graph squared source error; it does not explain the native subgroup failure. Normalization remains explicit.

HT/tensor-train/tree changes and arithmetic-DAG reuse address representation cost; they do not by themselves select unique variables. Weighted-automaton/Hankel realization would require a specified sequential input-output family and observable state equivalence; the current static four-output quadratic object does not supply that data. A full-support Gaussian gives positive norm to every nonzero polynomial, so changing from a positive coefficient metric to Gaussian functional error changes geometry but does not itself identify a unique internal circuit. These are scope arguments, not extra recovery theorems.

## Organization and decision

The continuation dossier remains the circuit authority; direct_tensor_match contains code/receipts. The requested overview now separates broad and selected targets and links fresh failures. Add current stability and consensus evidence to the dossier and one follow-up report, avoiding another copy of the whole trajectory. Reused the native bootstrap auditor with a version argument rather than a second scorer. Managed runner is live, current research jobs terminal; no new GPU job is required for this CPU distinction.

Weights-first work hands circuit work an operational choice: interpret stable downstream-defined functions or seek stronger interventions identifying internal groups. Circuit work must eventually test selective semantics and upstream closure. Current finding argues against labeling individual learned products as circuits. It does not invalidate the product-count saving or justify reducing the full goal to compression.
