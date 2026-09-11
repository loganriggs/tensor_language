# All-donor first-order alignment

V2 interchange passes A/B, but C misses for branch 0 on FineWeb. Its four fixed donor draws and descriptive fixed-donor bootstrap leave a possible sampling explanation. Preserve that miss. This follow-up averages the linearized intervention over **all** eligible other documents, then resamples paired documents and recomputes the statistic instead of conditioning on four donor assignments.

Within a family/domain stratum of $n$ documents, amplitude $a_i$ and loss sensitivity $g_i=\nabla_h\ell_i^\top w$ give

$$
\frac{1}{n(n-1)}\sum_{i\ne j}g_i(a_j-a_i)
=-\frac{n}{n-1}\left(\overline{ag}-\bar a\,\bar g\right).
$$

This is minus the sample covariance. Use the two unchanged 128-document panels, original factors and full native tail. Extract amplitudes and sensitivities with the immutable V2 analytic kernel; load only the unembedding. Baseline tail batches remain eight. Store the small scalar tables so CPU uncertainty analysis is reproducible without additional GPU work.

Average the three family expectations and weight each domain by its document count. Bootstrap documents with replacement within domains, using the same drawn document indices for every family and branch. Recompute each covariance from each resample; 2000 replicates, seed 5801. This addresses the previous fixed-donor dependence, but remains a small, previously inspected panel and a first-order local statistic. It does not establish exchangeability, task semantics, exact full-dose effects, or new OOD confirmation.

- A: baseline CE replay at most $10^{-5}$; old four-donor linear document effects replay at most $10^{-7}$; explicit all-pairs versus covariance identity at most $10^{-12}$; finite/source/shape checks.
- B: every single-branch all-donor mean is at least 0.001 nats, on both panels.
- C: each single-branch bootstrap lower 95% bound exceeds zero, on both panels.

Negative or uncertain results delimit these branches; no threshold or output-family changes. Price: 768 native tail rows, zero body forwards; CPU all-pairs and bootstrap analysis. No fitted coefficients, means substituted into the model, or circuit promotion.
