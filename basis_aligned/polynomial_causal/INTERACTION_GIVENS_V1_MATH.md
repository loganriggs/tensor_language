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
