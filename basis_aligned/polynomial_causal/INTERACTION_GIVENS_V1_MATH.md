# Learning a sparse interaction with cheap coordinate transforms

13 September 2026, 06:31 UTC. The first learned-frame pilot fails its storage-gain criteria. An executed change of objective produces a small positive gain, but still misses its registered 5% bar. These are bounded optimization pilots, not converged negative results.

## Why this family is different

The earlier orthogonal-output rank 32 block fit already has locally converged starts. Its input-unfolding bound excludes high coefficient fidelity at that rank budget. Repeating it is not the next experiment. Conversely, a full residual-space orthogonal frame costs 1152² coefficients and erased earlier sparse-core savings.

Here the residual transform is a sequence of two-coordinate rotations. Each creates two linear combinations of existing coordinates and can be executed without a dense 1152×1152 matrix. This permits a structured arithmetic graph even when the overall input map is full rank. We retain the existing output/head HOSVD frames and fit only the residual transform from weights.

Givens-coordinate optimization is an established approach to orthogonal problems [Shalit and Chechik2014](https://proceedings.mlr.press/v32/shalit14.html). Representing transforms with a short sequence of rotations makes application cost depend on the number of factors; generic orthogonal maps need not admit cheap approximations [Frerix and Bruna2019](https://arxiv.org/abs/1905.05796). These papers motivate our representation. Their results do not establish global recovery for this custom objective or finite random schedule.

## First objective: concentrate fourth powers

For two flattened residual-coordinate slices $x,y\in\mathbb R^{1536}$, apply

$$
x'=\cos\theta\,x+\sin\theta\,y,\qquad
 y'=-\sin\theta\,x+\cos\theta\,y.
$$

Their total fourth power is

$$
F(\theta)=\mathrm{constant}+C\cos 4\theta+S\sin 4\theta,
$$

where

$$
C=\frac 14\sum_j(x_j^4+y_j^4-6 x_j^2 y_j^2),\qquad
S=\sum_j x_jy_j(x_j^2-y_j^2).
$$

Thus $\theta=\tfrac 14\operatorname{atan 2}(S,C)$ is an exact pair maximizer. Disjoint pairs can be updated together. Orthogonality preserves coefficient norm and the relation between entry thresholding and reconstruction error.

Three random pairing schedules each executed 40 rounds of 576 pair rotations. Planted rotated one-row support recovers to 3.71 e-17; all objective and norm checks pass. This takes 3.99 CPU seconds. Fourth-power objectives increase 8.03–8.33 fold, yet at 10% coefficient error only 2633–3635 fewer coefficients can be dropped from the required retained set. Each trajectory adds 23,040 rotations. Pricing two uint 16 IDs plus FP 32 sine/cosine (12 bytes/rotation) overwhelms the coefficient savings. Every declared nonzero checkpoint is more expensive than the initial frame. Both registered gain criteria fail.

The last-round objective changes remain 0.4–1.0%, so this is not convergence. The stronger diagnostic is objective mismatch: concentrating the largest entries can substantially increase fourth powers without reducing the broad coefficient tail we need to preserve.

## Executed alternative: penalized reconstruction with a rotation price

For fixed threshold $\lambda$, the optimized entry-selection objective is

$$
\min_Y\|X-Y\|_F^2+\lambda\|Y\|_0
=\sum_j\min(X_j^2,\lambda).
$$

Here $\lambda$ has squared-coefficient units. Select it from the current global cutoff for 10% coefficient error. Search 17 pair angles including zero. A rotation is accepted only if it reduces this capped-energy objective by more than $3\lambda$: its 12 stored bytes equal the price of three FP 32 coefficients. This is a local penalty tradeoff, not a guarantee of actual global storage improvement after the threshold changes. Therefore we separately measure actual retained counts and total bytes at every checkpoint.

Three schedules each execute 10 rounds. All norm and fixed-threshold accepted-step checks pass. The best declared checkpoint stores 1,122,633 coefficients plus 2333 rotations, for 4,805,824 bytes versus 4,896,936 initially. That is **1.86% additional storage saving**, or 32.10% versus denseT. It misses the preregistered 5% improvement criterion. The three final gains are close; none is claimed stationary. The run takes 9.26 CPU seconds.

This countercheck changes the interpretation of the first null: the objective matters, and cheap learned transforms can pay for themselves. It does not yet give a large improvement, node sparsity, stable factor identification, or evidence for extending this greedy schedule indefinitely. Regional validation has not been rerun for these new frames, so the preceding frozen sparse operator's behavioral results cannot be transferred to them.

A next refinement should optimize actual reconstruction/graph cost and judge marginal gain per added rotation, or change to shared block/product topology. Do not use the impressive fourth-power increase as evidence of useful circuit simplification.

[Fourth-power receipt](INTERACTION_GIVENS_V1_RESULT.json) · [Objective mismatch audit](INTERACTION_GIVENS_SURROGATE_AUDIT_V1_RESULT.json) · [Capped-energy receipt](INTERACTION_GIVENS_CAPPED_V1_RESULT.json) · [First solver](interaction_givens_v1.py) · [Direct-objective solver](interaction_givens_capped_v1.py).

## Retained-circuit validation, 13 September 11:57

The existing winner was reconstructed deterministically rather than searched again: seed61332, round10, exactly 1,122,633 retained entries and 2,333 rotations, totaling 4,805,824 nominal bytes. [Reconstruction](reconstruct_givens_candidate_v1.py) checks those recorded counts before validation. The coefficient tensor is reconstructed in FP64 for this screen; packed FP32 rotation execution and runtime remain untested.

[The retained-effect check](INTERACTION_GIVENS_RETAINED_EFFECT_V1_RESULT.json) uses the existing 120 regional mixed-input ports, with native RMS factors and final softcap. Its reference is the effect of the mixed term itself: original margin minus the margin after removing that term. Candidate and baseline are compared on the same five groups of24, with a 10% own-effect-error bar and no material sign reversals.

| Group | Original sparse-frame effect error | Learned rotations, original smaller artifact | Learned rotations, matched budget |
|---|---:|---:|---:|
| 0 | 7.35% | 10.93% | 11.62% |
| 1 | 3.63% | 4.90% | 4.01% |
| 2 | 8.12% | 9.59% | 8.12% |
| 3 | 6.63% | 7.46% | 6.60% |
| 4 | 2.72% | 4.94% | 4.59% |

The original smaller learned candidate fails group0, with no material sign reversals. A storage mismatch could have unfairly penalized it, so [the matched-budget countercheck](INTERACTION_GIVENS_MATCHED_EFFECT_V1_RESULT.json) keeps the rotations fixed and spends the saved91,112bytes on22,778additional coefficients. It now uses the same4,896,936byte budget as the baseline. Coefficient error improves from approximately10% to9.423%, but group0 effect error rises to11.62%, still failing. No material sign reversals occur.

Thus the additional storage saving was real, but this learned frame does not preserve the retained interaction as reliably as the earlier frame. Lower coefficient error at matched bytes does not rescue the behavioral criterion. This is a limited failure of the recorded greedy trajectory, not an optimal sparse-frame or structure-absence result. The native context, input-port and historical-panel limitations remain; no new OOD or full-head preservation claim is made.

## Why did the effect prediction worsen?

[A weight-only error partition](INTERACTION_GIVENS_OUTPUT_PARTITION_V1_RESULT.json) tests whether the learned frame merely sacrifices token differences to preserve shared content. It does not: common-part error improves from6.12% to5.80%, and difference error improves from21.46% to20.17%. Every one of the six individual contrast coefficient errors improves. The orthogonal error partition checks within $1.21\times10^{-18}$ of total target energy. Thus reweighting common versus difference content alone is not an established explanation for this failure.

[The input-port check](INTERACTION_GIVENS_PORT_ERROR_V1_RESULT.json) locates the degradation before the final nonlinear readout. In group0, designated-pair mixed-numerator error rises from **6.11% to10.17%**; after native RMS scaling but before softcap, it rises from **7.09% to11.53%**. The final-effect comparison was7.35% versus11.62%. The deterioration is already present in the bilinear contraction on the actual retained inputs, so neither RMS weighting nor softcap alone causes it.

The useful conclusion is about the approximation target: less total or contrast coefficient error can still put more error along the inputs generated by this circuit. These diagnostics do not fit those inputs or authorize a data-guided replacement. A weight-first successor should investigate folding the actual producer constraints into the mixed operator, with independently frozen validation, rather than assuming an output weighting adjustment will repair the miss.
