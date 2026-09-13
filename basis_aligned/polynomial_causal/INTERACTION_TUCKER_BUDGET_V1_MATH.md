# Setting 2: dense Tucker cannot meet the current storage/error target

13 September 2026. The head17.2 → MLP17 → selected-output mixed operator is $T\in\mathbb R^{12\times1152\times128}$. The existing rotated sparse representation uses 4,896,936 nominal bytes at approximately 10% coefficient error. It prunes output edges but leaves almost every input-product node active. This screen asks whether a dense Tucker core could instead remove whole reader/product nodes at the same budget.

For output, residual and head ranks $(o,r,h)$, represent

$$
\widehat T=G\times_0 A_o\times_1 A_r\times_2 A_h.
$$

The core costs $orh$ scalars and adapters cost $12o+1152r+128h$. FP32 storage is therefore $4(orh+12o+1152r+128h)$ bytes, before metadata. Initially leaving the output mode unchanged gives $4(12rh+1152r+128h)$.

Any such tensor has unfolding ranks at most $(o,r,h)$. Let $e_j(k)$ be the discarded squared singular-value energy of mode $j$ after rank $k$, divided by $\|T\|_F^2$. Then necessarily

$$
\frac{\|T-\widehat T\|_F}{\|T\|_F}
\ge \sqrt{\max\{e_0(o),e_1(r),e_2(h)\}}.
$$

This lower bound applies to arbitrary factors at those multilinear ranks, including nonorthogonal factors. Optimizer convergence cannot evade it. It is a coefficient-error bound, not a behavioral impossibility result.

[The residual/head enumeration](INTERACTION_TUCKER_BUDGET_BOUND_V1_RESULT.json) finds no feasible pair that could reach 10%. Its best lower bound is **29.93%**, at ranks $(r,h)=(509,102)$ and 4,889,760 bytes. This check takes 0.86 CPU seconds.

The strongest immediate objection is that outputs themselves may compress. [The all-three-mode countercheck](INTERACTION_TUCKER_THREE_MODE_BOUND_V1_RESULT.json) includes the output adapter and enumerates every feasible rank triple. The best lower bound improves to **25.72%**, at $(o,r,h)=(9,569,108)$ and 4,889,952 bytes. Still no triple could reach 10%. This check takes 0.49 CPU seconds.

Individually, 10% coefficient error requires at least ranks **$(12,869,125)$**. In particular, output rank11 alone has 11.83% error, so it cannot satisfy the target. Even these individually necessary ranks require at least **9,282,928 bytes** for a dense Tucker representation—more than the 7,077,888-byte original dense tensor. Simultaneously achieving 10% could require larger ranks still.

## What this rules out—and what remains open

Do not launch a large HOOI or other dense-core Tucker fit at this budget: the representation cannot meet the stated coefficient target. This is stronger than another failed local optimization and costs little to establish.

It does not rule out a **sparse core**, overlapping/block-term factors, arithmetic sharing, cheaper structured adapters, or a producer-constrained target. At the individually necessary ranks above, dense adapters alone cost 4,068,928 bytes, leaving only 828,008 bytes for a sparse core and its support representation under the current budget. That is a concrete, demanding budget for a sparse-core successor; sparsity must be substantial enough to pay for the adapters.

Existing sparse interaction behavioral failures on broader native-head ports remain unchanged. This screen neither repairs them nor rejects the retained-interaction result. It redirects the node-sparsity search toward structured/sparse cores or composed producer restrictions instead of an infeasible dense-core model.
