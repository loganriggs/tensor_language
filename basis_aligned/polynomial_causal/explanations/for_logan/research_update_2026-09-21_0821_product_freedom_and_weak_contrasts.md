# Product identities can change cheaply; grouping exposes a weak missing contrast

21 September 2026,08:21 UTC. Follow-up to [random-start discovery](research_update_2026-09-21_0811_random_discovery_and_identity.md).

**We now have a constructive example of the identification problem: a finite arithmetic rewrite substantially changes important product nodes while barely changing the fitted function. Grouping products into larger quadratic features is more stable, but one weak contrast remains poorly recovered.**

These are coefficient-space and opened-state findings. They do not establish semantic circuits or fresh behavioral adoption.

## A selective rewrite succeeds where forced pairing failed

Previously we paired all256shared products by similar output directions and rotated each pair45degrees. A few poor forced matches dominated the resulting13.51%function change.

This time, those matches were optional. We ranked the same disjoint pairs by their individual squared function change divided by affected atom energy, then computed the **exact combined error**, including interactions between pair residuals. The largest prefix under a fixed2%coefficient-change budget contains79pairs.

| Measurement | Selective rotation |
|---|---:|
| Changed products |158of256shared products|
| Change relative to the parent fitted tensor |**1.86%**|
| Affected pre-cancellation atom energy |**50.33%**|
| Median best-matched atom cosine after the rewrite |**0.520**|
| Component1/2/3 value errors |2.66%,2.59%,11.94%|

The parent value errors were2.65%,2.48%,11.94%. Product count and coefficient storage are unchanged:512source products including the private branch,897,804floating coefficients. Exact implicit and dense error calculations agree within1.5e-16.

Here an **atom** is one product together with its output effect. Atom energy sums each atom's squared coefficient norm before cancellation; it is not causal importance. The1.86%figure measures change from the parent approximation, not error against the original model.

## Those products are not simply disposable

As a negative control, we physically deleted the same158products, retaining the exact original centered mean and linear terms. This reduces the program to354source products and533,140floating coefficients, but changes the fitted tensor by **70.81%**, about38times the rotation change. Component errors rise to15.22%,8.82%,11.94%, failing the fidelity limits.

Thus the affected terms carry substantial computation. Their particular product representation nevertheless has meaningful approximate freedom.

For equal output weights, the underlying identity is exact:

$$
y=w\left[(l_1^\top z)(r_1^\top z)+(l_2^\top z)(r_2^\top z)\right].
$$

Applying the same orthogonal two-column rotation to the left and right factor matrices preserves their summed product. Similar output weights can permit an approximate version, but the error must be measured; the earlier forced-pair failure shows why cosine alone is insufficient.

This demonstrates an approximately function-preserving rewrite at the same cost. It does not prove that the exact original tensor has multiple identical minimum-cost programs.

## Larger quadratic features are more stable—but not uniformly

The four normalized source reads form a tensor T with four output coordinates and two input indices. We diagonalize its output Gram matrix, obtaining orthogonal output directions U and quadratic features

$$
G_g=\sum_o U_{og}T_o,\qquad
h_g(z)=z^\top G_g z,\qquad
q(z)=Uh(z).
$$

Each h_g is a combination of existing products. This is an exact regrouping of a fitted tensor, not a reduction to four products. Its basis depends on the chosen coefficient metric and output eigenvalue gaps.

We independently grouped four warm-start fits and four random coefficient-only fits. Relative to the parent:

- The first three quadratic groups have cosine at least0.996across fits.
- The fourth has cosine0.948–0.955, failing the registered0.99limit.
- Output directions themselves are almost identical across fits.

The original target assigns only **0.186%of coefficient energy** to the fourth output contrast. The parent approximation's corresponding quadratic feature has cosine **0.858**with that original feature. This is a weakly weighted target, not a nearly degenerate output eigenspace: the output eigenvalues have substantial relative gaps.

Restart stability and original-function accuracy are different tests. The first three parent features have original-target cosines0.9975,0.9962and0.9886; even the third does not clear0.99fidelity.

## Small coefficient energy does not justify dropping the contrast

Removing the weakest contrast from the original quadratic reads causes **7.93%and6.97%component errors** on the opened states. If we remove only its centered quadratic remainder while preserving its original mean and linear terms, the errors are1.40%and0.64%.

This distinction matters because our fitted programs preserve those lower-order contributions. Their good scalar scores can coexist with a poorly recovered quadratic contrast. Neither result licenses silently removing that contrast from the target or declaring it irrelevant OOD.

A next structural comparison is to give low-energy output contrasts more weight during fitting. At the current capacity, necessary coefficient-error lower bounds rise from6.40%without output reweighting to7.43%with partial whitening and9.65%with full whitening. These are different metrics; the percentages cannot be ranked as the same error. The bounds quantify a capacity tradeoff rather than ruling out stable groups.

## Evidence and scope

All new analyses ran on CPU in FP64. Native scalar diagnostics reuse448opened sites, with native earlier and later inputs supplied. No new native-model forward or fresh document panel was used. The original private third component and its known fresh-panel failure remain part of the target.

- [Selective rotation curve and checks](../../direct_tensor_match/SELECTIVE_PRODUCT_ROTATIONS_V1.json).
- [Physical deletion control](../../direct_tensor_match/ROTATION_DELETION_CONTROL_V1.json).
- [All four canonical groups across fits and the original target](../../direct_tensor_match/SOURCE_CANONICAL_GROUPS_V1.json).
- [Weak-contrast removal and output-reweighting bounds](../../direct_tensor_match/WEAK_SOURCE_CONTRAST_V1.json).

The canonical-group gate was specified in the initiating tool call before execution; its attempted board append failed because of a quoting error and was repaired before results were inspected. That correction remains in the board. The all-groups stability gate fails as written.
