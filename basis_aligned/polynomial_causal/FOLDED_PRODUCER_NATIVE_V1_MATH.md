# Folded producer structure and its native behavioral failure

12 September, results through16:32UTC. This follows the [four-reading source interface](REGIONAL_COMPETING_CUES_V1_MATH.md), [failed shared cubic fit](FOLDED_PRODUCER_CUBIC_NATIVE_V1_RESULT.json), and [requested report section10](explanations/for_logan/research_update_2026-09-12_1536_interaction_decomposition.md#10-latest-follow-up-folding-the-regional-readers-into-their-upstream-attention-producers). The report's pending behavioral test is now complete; its timestamped account remains unchanged.

## Native verdict

The shared linear parent recovered by two cubic source fits is almost exactly head13.0's leading folded value reader. Retaining that reader with full native QK1×QK2 captures63–65% of this head's folded coefficient energy, but it **fails native effect fidelity and the registered meaningful-transfer threshold**.

The managed cache executed12 native batches in1.38seconds, with producer-fold error<=3.14e-7 and bit-exact replay of saved contextual states. Its 685KB artifact holds the contribution to four downstream readings from attention8+9+13, head13.0, and its rank1 value part. CPU interventions retain recipient query, projected key norms, first-token lookup, other producers and downstream background. They swap donor contributions divided by recipient residual RMS into all four readings. This is a conditional producer-to-reader edge intervention, not recursive replacement of a native head.

| City assignment / clause order | Full group mean directed transfer | Whole head13.0 | Euclidean value component | Consumer-weighted component |
|---|---:|---:|---:|---:|
| Original / editor first | .16969 | −.01051 | .00428 | .00508 |
| Original / editor second | .13859 | −.00888 | .00333 | .00392 |
| Reversed / editor first | .24815 | .00919 | .00323 | .00398 |
| Reversed / editor second | .19902 | .00847 | .00331 | .00407 |

Transfer is the change in British-minus-American token margin, signed toward the donor cue and averaged across24 directed swaps per cell. Each reverse direction is included; these are not24 independent semantic concepts.

The Euclidean component preserves the sign on95/96 swaps, with unrelated transfer below the registered50%relative ceiling, but provides only1.3–2.5%of the full producer-group transfer rather than the required10%. Its full-head effect error is59–141%, rather than<=10%. The complement opposes the selected component for the original city assignment. Component-plus-complement effect nonadditivity is only.24–.41%of full-head effect norm. Thus suffix nonlinearity does not explain away the disagreement. The head's city-pair-dependent sign does not establish a stable regional semantic unit.

[Registered bars](STRUCTURED_PRODUCER_NATIVE_V1_PREREGISTRATION.md) · [native cache](STRUCTURED_PRODUCER_CACHE_V1_RESULT.json) · [behavioral result](STRUCTURED_PRODUCER_EFFECT_CPU_V1_RESULT.json).

## Executed negative-result audit: coordinate dependence

The frozen downstream source polynomial is

$$
F(q,f)=\beta_0(q)f_0f_1f_2+\beta_1(q)f_0f_1f_3.
$$

For diagonal nonsingular S=diag(s0,...,s3), replacing f by Sf and dividing beta_a by s0*s1*s(2+a) leaves the function identical. The compiled executor realizes this by rescaling readings and compensating the two dual rows.

An ordinary leading singular component of the four-output value matrix M is not invariant under this freedom. After rescaling one reader100-fold, the whole branch replays bit-for-bit, but the leading source-reader cosine can drop to.00577 and apparent retained energy rises above99.8%. This is a limitation of treating arbitrary intermediate coordinates as a Euclidean output space. It does not invalidate the registered fit in its chosen coordinates or establish that natural unit-reader coordinates are useless.

[Gauge audit](FOLDED_PRODUCER_READER_GAUGE_V1_RESULT.json) implements this counterexample on the actual frozen component.

## A weights-only invariant alternative, solved exactly

Define a consumer pullback metric

$$
H=\mathbb E\left[J_fF(q,f)^T J_fF(q,f)\right].
$$

Here J is the derivative of the downstream polynomial output with respect to the four readings. The expectation is over independent isotropic Gaussian query/current inputs, uniform complete-vocabulary first-token initialization, and32 relative positions. There is no language-data or task-label fitting. This choice is an explicit distributional assumption; real contextual correlations, finite-change nonlinearities, final-suffix sensitivity and QK normalization gates are omitted from the discovery metric. Native validation retains the actual gates.

Write f=Cx+t, where x is standard Gaussian and t ranges over the full token lookup. Let Sigma=CC^T. The required fourth moments are the token fourth moment, six products of Sigma with token second moments, and the three Gaussian covariance pairings. Query-dependent beta functions are quadratic, so their output Gram also uses exact fourth moments. These Gaussian pairings follow [Isserlis's original moment formula](https://academic.oup.com/biomet/article-abstract/12/1-2/134/193428). We implement the finite token sum and independent Gaussian parts directly.

For a fixed metric, solve

$$
\min_{\operatorname{rank}(\widehat M)\le1}
\|H^{1/2}(M-\widehat M)\|_F^2.
$$

When H is positive definite, this is ordinary truncated SVD after the invertible left transformation H^(1/2), a special separable weighted low-rank problem rather than general elementwise weighted fitting. See [Markovsky's treatment of weighted low-rank approximation](https://imarkovs.github.io/book/book2e-2x1.pdf). If v is the leading right singular vector of H^(1/2)M, the solution is Mhat=Mvv^T. No iterative convergence issue remains for this fixed matrix problem.

Under the exact re-encoding, H'=S^(-T)HS^(-1), M'=SM, so

$$
M'^T H' M'=M^T H M.
$$

The selected source direction is invariant when its leading eigenvalue is isolated. The equivalent four-output projector P=Mhat M^+ is generally oblique; row-valued cached contributions must multiply P^T. We charge the full native QK and state dependencies as before.

Independent controls pass: metric covariance error<=5.63e-16; back-transformed selected-map error<=3.54e-13; Monte Carlo checks of the exact fourth moments and query Gram disagree by1.50%and.784%, respectively. H has eigenvalues .175,.182,.199,.444 after trace normalization. Thus the original natural coordinates were only moderately anisotropic under this particular consumer metric.

## The invariant metric does not rescue this candidate

The weighted source direction has cosine.99916 with the original one. It gives96/96 signed swaps in the expected direction but only1.6–3.0%of the producer-group transfer. Full-head fidelity error remains51–148%. Both B/C predictions fail under their unchanged thresholds. The new method solves an actual coordinate-dependence problem; the behavioral miss survives it.

[Metric controls](CONSUMER_PULLBACK_V1_CONTROL.json) · [frozen comparison](CONSUMER_PULLBACK_V1_PREREGISTRATION.md) · [native weighted result](CONSUMER_PULLBACK_EFFECT_CPU_V1_RESULT.json).

**Narrow conclusion:** the leading folded value component of this coefficient-dominant producer head is not a sufficient regional producer at the registered boundary. Neither a high coefficient fraction nor gauge invariance establishes circuit significance. This does not rule out more complex shared QK/value structure, other heads, or sparse mixed producer–consumer paths. The next discriminating object is the exact cubic interaction among producer contributions and background, before treating any one producer direction as the unit of computation.

## 16:36 — Exact interaction paths carry the group effect

Write the four downstream readings as f=B+H+O: B is the recipient background after subtracting the attention8/9/13 group, H is wholehead13.0's contribution, and O is the rest of that group. Contributions include recipient RMS scaling. For each child, expanding the three ordered reader slots and collecting equal source-group multisets gives ten paths:

$$
B^3,\ B^2H,\ B^2O,\ BH^2,\ BHO,\ BO^2,\ H^3,\ H^2O,\ HO^2,\ O^3.
$$

These symbols mean the sum of all assignments to the parent0, parent1 and child reader slots with that multiset. For example BHO includes six assignments; it is not six times one ordered product when the reader roles differ. All terms retain the original query consumer and normalization. The decomposition is an exact algebraic diagnostic, not necessarily a cheaper implementation than the original two shared cubic features.

The summed writes replay within1.03e-7 and the full group-swap effect exactly matches the previous CPU test. The seven mixed paths reproduce that effect within**.16–.29%** in each of the four cells (registered10%bar passes). But the three paths containing both H and O carry only**.8–1.9%**of full effect norm, missing the registered10%cross-producer contribution threshold. This is mainly **background–producer mixing**, not strong cooperation specifically between head13.0 and the other selected producers.

Keeping only B^2H+B^2O incurs**11.8–14.1%**effect error. BO^2 is the largest remaining contribution, with mean directed effect .0160–.0224. Individual path margin effects sum to the full effect within .048–.108%, although additivity was not assumed by the test. Pure producer cubes are small under this particular partition and panel; this does not establish that all cubic paths are unimportant or that the background is independently generated.

The mixed-path pass is a conditional local result and may partly reflect small producer contributions relative to background. No behavioral fit selected these ten terms, but the producer group itself came from earlier behavioral tracing. A broader unsupervised decomposition cannot be inferred from this one cut. [Exact executor and all cells](PRODUCER_INTERACTION_CUT_CPU_V1_RESULT.json).
