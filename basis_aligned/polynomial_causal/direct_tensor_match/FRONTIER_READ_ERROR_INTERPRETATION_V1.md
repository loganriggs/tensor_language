**The remaining component-three error is primarily first-order read error and reinforcement**

21 September 2026, 15:21 UTC. This is an explanatory analysis of the same 32 opened code files, not new independent validation. The frozen graph, covariance baseline, tokens and donor maps are unchanged.

Write the true component as $ab$, where

$$
a=(h^\top A_3-\tfrac12q_a)/s-\alpha_3,
\qquad b=q_b/s-\beta_3.
$$

For fitted-read errors $\delta_a=\hat q_a-q_a$ and $\delta_b=\hat q_b-q_b$, the component error is exactly

$$
\hat\phi-\phi=
\underbrace{-\frac{\delta_a b}{2s}}_{t_1}
+\underbrace{\frac{a\delta_b}{s}}_{t_2}
-\underbrace{\frac{\delta_a\delta_b}{2s^2}}_{t_3}.
$$

We retain the full signed Gram matrix $K_{ij}=\sum_n t_i(n)t_j(n)$. Total squared error is the sum of **all** entries, including cross terms. For donor-change errors, subtract paired term vectors before computing their Gram matrix.

Source replay error is 1.30e-7; error-term replay is 2.35e-13; aggregate energies reproduce the previous scalar diagnostic within 1.48e-14 relative error. The managed run took 2.65 seconds. All registered predictions pass, including the prediction that the second-read diagonal error increases in both failing cohorts.

For natural/spaced-word sites, the graph's excess squared error over the baseline splits into approximately 44.0% from the second-read diagonal and 48.5% from the first/second signed cross term. For hybrid/spaced-word sites these contributions are 43.1% and 37.5%, respectively. These are exact algebraic energy differences, not causal attributions; signed contributions can in general be negative or exceed 100%.

The product-error term itself has only 1.30% and 1.16% of the total error norm. Thus the hypothesis that a large product of the two read errors drives the failure is unsupported. Improving the second-read contribution and its correlation with the first-read error is better motivated than adding capacity specifically for that product-error term.

Diagnostic exact replacement of the second read would leave 42.1% of the graph's natural error norm and 47.7% of its hybrid error norm. Exact replacement of the first read leaves 83.6% and 81.0%. These replacements are expensive oracles, not proposed smaller programs.

**Mean-error follow-up actually performed**

The squared mean accounts for 33.5% of graph natural error energy and 31.4% of hybrid error energy in the failing cohort, versus 14.2% and 12.9% for the baseline. That motivates a descriptive cross-panel check rather than assuming a constant bias.

Using each panel's all-token natural mean error as an offset on the other panel produces graph/baseline error ratios of 0.930 and 0.975 on panel two's spaced-word natural/hybrid sites, and 0.932 and 0.887 in the reverse direction. **Both programs receive their own estimated offset.** The same constant cancels from donor-change effects, which remain unchanged. Cohort-specific offsets also yield ratios below 1.10 in these four checks. These are post-hoc scalar analyses, not native endpoint tests and not a repair of the frozen candidate. Baseline correction can itself worsen a subgroup, as it does in the reverse all-token calibration, so the relative ratios alone cannot establish adoption.

A useful next candidate is a component-level constant calibrated only on the original fitting states, applied equally to graph and baselines. It differs from the existing mean/affine corrections to each quadratic read: the component is a product of normalized reads. Such a candidate must preserve derivative and donor-change semantics, include its coefficient/addition cost, and be evaluated for actual endpoint effects and all cohorts. These opened panels cannot become fresh evidence again. If calibration on the original states fails to transfer, reject the constant explanation as a practical repair and return to direction/correlation changes; do not fit offsets to the failing cohort and call that generalization.

This does not close extraction, semantic identity, general graph search or the broader folded-model objective.

[Managed results](FRONTIER_READ_ERROR_V1.json) · [Signed accounting and exact-read oracles](FRONTIER_READ_ERROR_AUDIT_V1.json) · [Cross-panel mean transfer](FRONTIER_READ_ERROR_MEAN_TRANSFER_V1.json).
