# The continuation feature matches a leading component of the original folded weights

21 September 2026, 03:56 UTC.

**The learned continuation feature is almost the best rank-one approximation to a precisely defined readout of the original folded operator.** In the calibration-covariance metric, its coefficient cosine with that optimal native component is 0.99995. This is not a global isotropic low-rank claim. This closes an important part of the gap between behavior of an approximate graph and structure in the original weights.

It does not reconstruct the entire readout: rank one leaves 31.5% relative coefficient error. Additional components improve native effect reconstruction, but the registered all-cohort fidelity test still fails. Folding the feature's inputs through the preceding MLP is exact, yet those upstream forms are not compact under an isotropic low-rank approximation.

## Which original computation did we test?

Let $w$ be the frozen learned output direction in QR-reduced coordinates. Let $K=S^\top S$ be the fixed vocabulary-centered output metric. Define an output reader

$$
q=\frac{Kw}{w^\top Kw},\qquad q^\top w=1.
$$

Then $wq^\top$ is the metric-orthogonal projection onto that direction. The direction was discovered through the earlier decomposition; once fixed, its native projection is unambiguous.

For centered intermediate inputs $n_c,m_c$, the original folded interaction is

$$
Y(n_c,m_c)=C[(Ln_c)\odot(Rm_c)+(Rn_c)\odot(Lm_c)],
\qquad C=R_UD.
$$

Its scalar readout is exactly

$$
t(n_c,m_c)=q^\top Y(n_c,m_c)=n_c^\top M m_c,
$$

$$
M=L^\top\operatorname{diag}(C^\top q)R
 +R^\top\operatorname{diag}(C^\top q)L.
$$

All matrices here come from the original trained last MLP, apart from the specified output reader. Centering leaves the original constant and single-input terms explicit outside this interaction. This is not the whole MLP or model.

## Was the previous learned feature a fitting artifact?

We compared its single product with the globally optimal rank-one approximation of $M$ in the same independent, centered calibration-covariance metric. This restricted problem is solved by an ordinary SVD of the weighted matrix; optimizer choice does not determine its optimum.

| Approximation of the native scalar readout | Relative coefficient error |
|---|---:|
| Frozen learned product | 31.469% |
| Optimal native rank one | 31.455% |
| Frozen product plus explicit remaining fitted terms | 7.253% |

The learned product is nearly optimal, and its tensor direction has cosine **0.999953** with the native rank-one solution. The first singular component contains **90.106% of squared coefficient energy**; the second contains 7.639%. The first singular value is 3.43 times the second.

Thus the registered 25% rank-one error target fails for a structural reason in this metric: even the global rank-one optimum cannot reach it. This is not a failure of Adam, Muon or initialization. Also, 90% retained squared energy means approximately 31.5% relative norm error—not 10% error.

The remaining fitted terms were included explicitly as a background rather than silently credited to this feature. They matter for reconstructing the complete scalar readout. No covariance directions were discarded in this calculation. Native saved-output replay was $3.32\times10^{-8}$, and scalar matrix/factor replay was $4.82\times10^{-15}$.

## The metric changes which structure is simple

We also solved the rank-one problem in ordinary isotropic coefficient Frobenius norm, using the same original scalar operator.

| Candidate | Isotropic coefficient error | Covariance-weighted coefficient error |
|---|---:|---:|
| Frozen learned product | 99.35% | 31.47% |
| Covariance-optimal rank one | 99.34% | 31.45% |
| Isotropic-optimal rank one | 95.51% | 98.99% |

The isotropic leading mode captures only 8.78% of squared coefficient energy. The strong 90.1% result therefore concerns the data-informed input metric, not all possible input directions equally. This helps explain why a useful feature on model states can coexist with poor global tensor-compression results. It does not establish that one can discard the covariance information without losing the discovered computation.

## What happens when the native projection is removed?

We removed $w\,t(n_c,m_c)$ from the native final residual through the existing QR mapping, then evaluated native final normalization and softcap. The comparison used the already evaluated 32-document confirmation panel; these are new interventions on reused documents, not another fresh-data result.

The exact original-weight projection shows the same behavioral pattern:

| Removed computation | Continuation CE added | Spaced-word CE added |
|---|---:|---:|
| Exact native projection | **0.13290** | **0.00017** |
| Native leading rank-one component | 0.12615 | −0.00021 |
| Earlier learned product | 0.12555 | −0.00021 |

Positive CE added means prediction damage. This supports interpreting the feature as a dominant continuation-related component of the specified native readout. It does not establish that the direction is the only possible circuit boundary or that every component sharing it has the same role.

## More components improve fidelity, but unevenly

We exported ranks 1, 2, 3 and 8 before native evaluation. The reference below is removal of the exact original-weight projection, not the previous approximation.

| Products | Coefficient error | Native effect error: all sites | Continuation sites | Spaced-word sites |
|---|---:|---:|---:|---:|
| 1 | 31.45% | 23.90% | 12.96% | 61.88% |
| 2 | 15.02% | 18.18% | 11.41% | 49.38% |
| 3 | 9.53% | 12.04% | 5.72% | **33.44%** |
| 8 | 4.76% | 8.18% | 5.32% | 29.74% |

The registered rank-three requirement was at most 15% native effect error in **every** cohort. It fails on spaced-word sites. The rank-one behavioral test and instrument checks pass.

Absolute sizes help interpret—but do not erase—the failure. Native centered-logit RMS is 0.9146 at continuation sites and 0.08576 at spaced-word sites. Rank-three error RMS is 0.05232 and 0.02868, respectively. A near-zero average CE change on spaced words does not imply accurate reproduction of their logit effects.

## Folding the feature inputs one layer further

For the native leading mode, write its readers as $a,b$. Let $z$ be the normalized input to MLP16, and let its scaled polynomial output be

$$
m_0=\lambda D_{16}[(L_{16}z)\odot(R_{16}z)].
$$

We folded both scalar source reads exactly:

$$
a^\top m_0=z^\top Q_a z,\qquad
b^\top m_0=z^\top Q_b z.
$$

Consequently, with last-MLP input $h$ and its original RMS denominator $s(h)$, the scalar feature can be written

$$
u=\frac{a^\top h-\tfrac12 z^\top Q_a z}{s(h)}-\alpha,
\qquad
v=\frac{z^\top Q_b z}{s(h)}-\beta,
\qquad \widehat y=wuv.
$$

The upstream normalization producing $z$, and the denominator $s(h)$, remain explicit operations. The preceding MLP's output bias is excluded consistently with the original source definition.

Both folded forms replayed the native weight contractions to below $4\times10^{-15}$ on synthetic normalized inputs. However, retaining 90% of their isotropic squared coefficient energy needs **480 and 485 eigen-directions**, respectively. Rank 64 still leaves approximately 80–81% coefficient norm error. This is not a small standalone upstream circuit under that metric.

That result does not rule out data-informed simplification or reuse of existing upstream products. It also does not eliminate the native input $h$, which itself contains earlier computations. The useful next question is whether the scalar inputs can be produced economically under a declared interface while retaining their behavioral effects—not whether we can merely relabel two dense quadratic forms as simple features.

## Evidence

- [Original-weight scalar grounding](../../direct_tensor_match/MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.json).
- [Isotropic/covariance comparison](../../direct_tensor_match/MIDPOINT_NATIVE_OBSERVER_METRIC_COMPARISON_V1.json).
- [Optimal modal exports and coefficient prices](../../direct_tensor_match/MIDPOINT_NATIVE_OBSERVER_MODES_V1.json).
- [Native modal removal results](../../direct_tensor_match/MIDPOINT_NATIVE_OBSERVER_EFFECTS_V1.json).
- [Absolute effect-size audit](../../direct_tensor_match/MIDPOINT_NATIVE_OBSERVER_EFFECT_AUDIT_V1.json).
- [Exact upstream fold and spectral costs](../../direct_tensor_match/MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.json).
- [Updated candidate dossier](../../direct_tensor_match/circuits/letter_continuation_group2.md).

The managed native job completed successfully; frozen-feature CE replay differed by $2.59\times10^{-9}$. CPU weight calculations used float64 and two Torch threads. Broad OOD behavior, upstream extraction, collateral controls and composition remain open, so the overall circuit goal is not complete.
