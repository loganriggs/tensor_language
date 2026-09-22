# Output balancing must change shared producers, not just independent readouts

22 September2026,01:07UTC. Consequence of the residual audit: features0–2 dominate value energy while features4–15 have high relative errors. Before treating a change of output metric as a remedy, distinguish fixed-feature readout fitting from shared-feature discovery.

Let phi(x) be an m-dimensional fixed feature dictionary, c_g its readout for output g, K=E[phi phi^T], and b_g=E[phi y_g]. For positive weights w_g, use

$$
J=\sum_g w_g\left(c_g^T Kc_g-2c_g^Tb_g+E[y_g^2]+\rho\|c_g\|^2\right).
$$

The normal equations give

$$
c_g=(K+\rho I)^{-1}b_g.
$$

Weights cancel separately for each g. Merely giving small outputs more weight cannot repair their predictions by refitting independent unconstrained readouts over the same dictionary. This remains true for any exact coefficient or Gaussian inner product defining K,b. With singular unregularized K, the set of minimizers is unchanged, although implementation choices can select different representatives.

If instead the ridge is NOT multiplied by w_g, the effective ridge becomes rho/w_g. Changed readouts then reflect changed regularization. A claimed metric benefit must separate that effect from changes to shared feature directions. Coupled readout constraints would be another genuinely different case.

When phi depends on learned parameters theta, the weighted residual gradients from different outputs add into the same producer gradient. Weights can then change how shared capacity is allocated. To retain the envelope-gradient identity, solve each readout under exactly the objective whose producer gradient is used. Teacher constants can be omitted for gradients but must not be confused with full relative reconstruction error.

Five CPU toy structures verify readout invariance to relative error<5e-16 with consistently weighted ridge: shared linear sum, shared product, dense rank-one quadratic, sparse pairs, and quartic cancellation. Producer-gradient cosines between uniform and balanced objectives are0.619,0.259,0.567,0.539,1.000 respectively. The final cancellation outputs are scalar multiples of the same polynomial, so weights only rescale the common gradient; this is a useful degenerate control, not an expected universal change. Unweighted ridge changes readouts substantially, as predicted. No native improvement has been measured by this control.

Decision: do not queue a fixed-feature output-balancing sweep expecting new representational capacity. If the currently running uniform-output producer fit fails individual component fidelity, an output-balanced producer fit is a distinct hypothesis. Use calibration-only or exact-metric output scales with a declared floor, freeze before fresh evaluation, and track both dominant-feature regression and small-feature fidelity. The present larger fresh panel evaluates only the five predeclared frozen candidates; later metric tuning must not turn it silently into a new held-out set.

Files: check_output_balancing.py, OUTPUT_BALANCING_CONTROL_V1.json. This is an algebraic control and experiment-design constraint, not a circuit-identification result.
