# Shared computations with sparse private corrections

13 September 2026. This is a weights-only screen of a different interaction graph, followed by an executed joint-fitting countercheck. It does not establish behavioral reuse or an adopted compression.

## Representation and literal price

Use the same selected folded mixed tensor $T\in\mathbb R^{12\times1152\times128}$, flattened as $X\in\mathbb R^{147456\times12}$. Reuse the32product groups of the converged ordinary output-subspace fit. Each product row has a shared component plus an independently sparse correction:

$$
x_e=Q_{g(e)}c_e+s_e+\epsilon_e,
\qquad Q_g\in\mathbb R^{12\times r}.
$$

The executable graph computes each input product, accumulates shared group features, writes through $Q_g$, and adds the product's selected private output edges. The correction therefore allows a product to share common output computations while retaining differences. Sparse residual entries do not remove the shared feature's input dependencies.

At rank $r>0$, shared codes and writers cost $4(147456r+32\cdot12r)$ bytes, assignments cost147456bytes, and the full residual support bitmap costs221184bytes. Each retained private coefficient adds4bytes. Allocate the remaining budget to these coefficients, giving exactly4,896,936nominal bytes. Rank0 is a native-coordinate sparse-only control without group assignments. Metadata, serialization, resident indices and execution time remain unmeasured; nominal equality is not adoption pricing.

For a fixed shared component, keeping the largest squared residual entries gives the globally optimal fixed-count correction in this native coordinate frame. The remaining residual energy gives the exact approximation error. It is not an optimality claim about the shared factors, group assignments or frame.

## Initial screen

[Code](interaction_shared_private_v1.py) and [receipt](INTERACTION_SHARED_PRIVATE_V1_RESULT.json). Conditional output PCA within the fixed groups, followed by residual selection, yields:

| Shared rank per group | Relative coefficient error at matched bytes |
|---|---:|
|0: native sparse-only|14.42%|
|1|19.07%|
|2|22.23%|
|3|23.97%|
|4|25.48%|
|5|25.18%|
|6|24.18%|
|7|20.31%|

All exact residual-energy and arbitrary-product readout checks pass; readout discrepancies are below $2\times10^{-14}$. The existing output/head-rotated sparse baseline achieves approximately10%error at this budget, including its adapters. The screen took1.90CPU seconds.

This sequential fit is not a fair negative verdict on joint sharing: it chooses the shared component without accounting for which errors private corrections will remove.

## Executed joint-fitting countercheck

For the strongest nonzero rank1 configuration, alternate

$$
(Q,c)\leftarrow\arg\min_{Q_g^TQ_g=I,c}
\sum_e\|x_e-s_e-Q_{g(e)}c_e\|^2,
\qquad S\leftarrow H_k(X-Qc),
$$

where $H_k$ retains the globally largest $k=984234$entries. The first conditional minimization is a separate rank1 PCA of $X-S$ within each fixed group. Both steps are exact for their declared block, and the objective must not increase. Group assignments remain fixed.

[Code](interaction_shared_private_refine_v1.py), [first receipt](INTERACTION_SHARED_PRIVATE_REFINE_V1_RESULT.json), [extended receipt](INTERACTION_SHARED_PRIVATE_REFINE_LONG_V1_RESULT.json). The100-update screen reached14.3260%error but missed its objective tolerance. A deterministic replay with a larger limit reached14.3258%after115updates, satisfying two consecutive relative squared-objective decreases below $10^{-7}$ in17.95CPU seconds. The original unfinished receipt is preserved. This stopping test is an objective plateau, not a global-optimality or full stationarity certificate; only one initialization was fitted.

Joint optimization substantially repairs the sequential fit, but gains only about0.092percentage points over native sparse-only and remains worse than the rotated sparse baseline. Thus the sequential failure exaggerated the problem, while the executed repair still does not justify adopting this configuration. No native behavioral panel was run for this candidate because it has not improved the coefficient/storage comparison.

## What changes next

Do not generalize this result to absence of shared/private structure. Fixed groups, native residual coordinates and one rank1 initialization remain restrictions. Nevertheless, the current evidence favors the simpler sparse-entry baseline over adding another bank of densely coded shared writers. A useful successor should expose producer-constrained input reuse or change how residual corrections are represented; more of the same grouping/PCA fit is low priority. Full behavioral reuse requires shared-producer and consumer-specific interventions after a candidate passes preservation, not just the existence of $Q_g$ in its formula.
