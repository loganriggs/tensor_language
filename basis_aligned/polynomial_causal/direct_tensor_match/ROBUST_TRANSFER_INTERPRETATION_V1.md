**The new robust feature does not solve text transfer**

The managed follow-up completed in10.40seconds on the same96opened documents, with exact-read interface replay9.06e-8. Both candidate graphs were compared against the same six strengthened baselines. This is a diagnostic reuse of data, not a fresh result.

| Interface | Graph | Absolute failures /144 | Distinct cells failing any relative comparison | Failed covariance comparisons | Failed isotropic comparisons |
|---|---|---:|---:|---:|---:|
| Supplied native residual | Previous |0|10|18|12|
| Supplied native residual | New robust feature |0|10|16|10|
| Generated residual projection | Previous |1|8|0|23|
| Generated residual projection | New robust feature |1|10|0|26|

Each baseline geometry has three width allocations; comparison counts therefore differ from distinct-cell counts. The generated-interface absolute failure is still FineWeb/panel2/component3/natural/continuation, improving only from16.94% to16.79% against15%. Component-three errors across cells range from5.0% better to4.9% worse than the previous graph. No candidate is accepted.

The positive mathematical result remains: a directed extra feature, unlike the random control, repairs the selected global worst-read constraint at small extra cost. That is not sufficient evidence for circuit fidelity or native transfer.

**Actual successor: how relevant is the global source bound?**

For true component factors $a,b$, fixed native RMS $s$, and read errors bounded by $\epsilon_a,\epsilon_b$, the triangle bound is

$$
|\delta\phi|\le
\frac{|b|\epsilon_a}{2s}
+\frac{|a|\epsilon_b}{s}
+\frac{\epsilon_a\epsilon_b}{2s^2}.
$$

On the original448states, the vector of these global bounds has norm3733times the previous graph's actual component-error norm and3004times the new graph's. By contrast, substituting the actual pointwise read errors in the same triangle expression gives factors1.245and1.255. Thus the dominant looseness comes from replacing state-dependent read errors with global ball extrema, not just from ignoring signed cancellation.

The new direction is not inactive: its empirical projection variance is1.230, compared with1for an isotropic unit direction. Its centered square has fitting-state RMS only0.0793%of its maximum over the full RMS ball; such a ball can place nearly all input norm along one direction. The new feature contributes read-change RMS135.3; refitting the older coefficients contributes only0.197. These values describe feature exposure and error geometry, not semantic identity or proof that adversarial directions are unreachable by text.

**Research decision**

Do not expand the global-ball target to every read and assume that will fix the functional goal. Retain the solver and directed-feature proposal as useful tools, but reject this candidate as a transfer repair. A subsequent fitting objective needs an explicit connection to the complete normalized computation and its input/context geometry, alongside a global coefficient safeguard. Prior Gaussian, empirical-moment, constant-offset and homogeneous-cubic failures remain constraints on that choice; merely renaming one of those losses is not a new hypothesis.

The full goal remains open: the current graph still receives native states or scalar context, has no established semantic feature identity, and has not passed the required predictive, selective-manipulation or reuse evidence for an extracted circuit.

[Same-baseline transfer audit](ROBUST_TRANSFER_AUDIT_V1.json) · [Native screen](ROBUST_TRANSFER_V1.json) · [Bound relevance and feature exposure](ROBUST_BOUND_RELEVANCE_V1.json).
