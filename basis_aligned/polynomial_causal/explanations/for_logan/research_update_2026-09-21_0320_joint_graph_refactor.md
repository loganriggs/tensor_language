# Joint graph refactoring: modest gains, and a corrected instability diagnosis

21 September 2026, 03:20 UTC.

**Jointly refitting products helped more than deleting products, but none of four tested edits met the registered improvement target.** A separate finding is useful for circuit identification: an apparent disagreement between near-best component decompositions disappeared with further optimization. We should not call every finite-step disagreement structural instability.

## A broader second-stage edit

Following the pairwise-merge limitation, we selected four disjoint neighborhoods of eight products from the frozen 512-product graph. Selection used the strongest remaining product as a seed and nearby input-product directions. Each neighborhood was represented exactly in a small coefficient tensor and refitted with six products:

$$
\sum_{k=1}^{8}w_k(a_k^\top n_c)(b_k^\top m_c)
\approx
\sum_{k=1}^{6}\widehat w_k(\widehat a_k^\top n_c)(\widehat b_k^\top m_c).
$$

Here $n_c,m_c$ are the centered midpoint and source inputs; each $w_k$ is a vector of output effects. Input directions and output sharing can change jointly. Constant and linear branches remain fixed.

The comparison was stronger than simply keeping the largest products: we enumerated all 28 six-of-eight subsets and refitted each subset's output writes. The target was **at least two neighborhoods with 25% lower squared error than that best subset**. The metric used independent centered calibration input covariances and vocabulary-centered output geometry. The teacher was the frozen approximation, not the complete native model.

## Initial results and capacity limits

Each neighborhood received 12 fits: Adam/Muon, learning rates 0.01/0.05, and a subset initialization plus two random initializations, each for 400 steps. A planted rank-three control recovered its target to approximately $10^{-10}$ relative norm error. Exact Gram-coordinate energy replay passed.

| Neighborhood | Best Adam/Muon squared error ÷ subset error | Lower bound ÷ subset error | Registered target possible under the bound? |
|---|---:|---:|---|
| 0 | 0.963 | 0.293 | Not ruled out |
| 1 | 0.965 | 0.919 | No |
| 2 | 0.999 | 0.998 | No |
| 3 | 0.995 | 0.980 | No |

The target ratio was 0.75. The bounds come from best rank-six matrix approximations of the tensor's unfoldings. They are necessary bounds, not guaranteed attainable tensor errors. Three neighborhoods cannot meet the target at six products in this metric, regardless of optimizer. Neighborhood 0 still warranted an optimization check.

## Independent solver audit

We fitted neighborhood 0 with alternating least squares (ALS), updating each factor using a linear solve while holding the others fixed. Sixteen initializations included eight subset/perturbed-subset starts and eight independent random starts. At 2,000 sweeps, the best squared-error ratio improved to **0.858**, a **14.2% reduction** relative to the strongest subset baseline. The local coefficient norm error is approximately 6.24%.

This remains a registered failure, not a passing result obtained by changing the threshold. It is nevertheless evidence that a joint edit can improve on subset selection. ALS received a different update budget; this does not establish a universal optimizer ranking. No edited full graph was adopted or evaluated in the native model.

## Why the component-identity check needed a redteam pass

Fourteen of the sixteen ALS runs were within 1% of the best squared error. We matched their six component tensors by optimal permutation and signed tensor cosine, which removes compensating factor-sign and scale choices. Most agreed closely, but the worst near-best run had minimum matched cosine **0.907** despite only about **0.096%** higher loss.

That initially looks like unstable components. We continued both this run and the best run on exactly the same target:

| Additional ALS sweeps | Minimum matched component cosine |
|---|---:|
| 0 | 0.90656 |
| 500 | 0.91364 |
| 2,000 | 0.94145 |
| 10,000 | 0.999999995 |

Their losses also converged. **This particular discrepancy was incomplete convergence.** We did not extend every restart, and this is not evidence of stability under changed documents, covariance estimates or task distributions.

The best fitted factor matrices have numerical full column rank. This is consistent with the sufficient Kruskal uniqueness condition for a fixed represented rank-six tensor, up to scaling and permutation. However, exact uniqueness does not ensure a well-conditioned approximation or semantic interpretation. The normalized output-factor matrix has condition number approximately 486. See the CP uniqueness discussion in [Kolda and Bader, section 3.2](https://www.kolda.net/publication/TensorReview.pdf).

## What changes next

The useful next identification test is sensitivity to independently estimated input metrics or document splits, after sufficient convergence. Repeating a short optimizer run would confound solver error with feature instability. The original six-product fit is a candidate computation to examine, not an identified linguistic circuit.

For graph search, these results favor testing broader edits where rank bounds leave room and using a strong subset-plus-refit baseline. They do not justify a general claim that Tucker/HT, sharing, or arithmetic circuits have failed.

Evidence: [initial screen](../../direct_tensor_match/MIDPOINT_JOINT_PRODUCT_REFACTOR_V1.json), [independent ALS audit](../../direct_tensor_match/MIDPOINT_JOINT_PRODUCT_ALS_AUDIT_V1.json), [component comparison](../../direct_tensor_match/MIDPOINT_JOINT_COMPONENT_STABILITY_V1.json), and [convergence correction](../../direct_tensor_match/MIDPOINT_JOINT_STABILITY_CONVERGENCE_V1.json). Saved factor artifacts preserve the examined computations. All work used CPU float64 and two Torch threads; there were no new model forward passes. A deterministic rerun saved ALS factors and is not counted as independent evidence.
