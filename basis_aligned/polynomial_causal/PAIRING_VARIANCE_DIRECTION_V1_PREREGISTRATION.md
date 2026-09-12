# Pairing-diagonal control variate: matched direction pilot

Keep the starting frame, fixed grade normalizations, seeds73220/21,512 probes
per grade per gradient, steps0.01/0.05/0.15 and independent2048-probe validation
seed73222 from REPEATED_METRIC_DIRECTION_V1. Replace the stochastic estimator
only. For N pairings and their error contractions e_p, estimate squared error as

$$
\|N^{-1}\sum_p e_p\|_G^2-N^{-2}\sum_p\|e_p\|_G^2
+\frac{\binom4k}{N}(F_k-R_k(P)).
$$

The last term is exact and differentiable on the orthonormal-frame manifold.
Its value need not be correct off that manifold, so check its derivative along
QR-retracted directions. The sampled remainder gradient is validated by the
exhaustive coefficient/tangent-gradient control. Degree1 has zero remainder.
No activation or text fitting. Two replicas provide a noisy variance comparison,
not a statistical confidence claim.

- A: native analytic-term tangent FD error<=1e-4, frame orthogonality<=1e-8,
  finite output; exhaustive toy identity/gradient checks<=1e-9.
- B: independent gradient cosine>=0.5 and squared replica-difference noise
  estimate <=0.5 times the matched naive pilot.
- C: some registered step improves independent balanced squared loss by>0.001
  and>3 paired standard errors, runtime<=240 seconds.

Null: replacing exact diagonal contributions does not make small-probe
optimization reliable. Preserve the original pilot miss. This fixed coefficient1
control variate is unbiased, but variance reduction is a hypothesis, not a theorem.
No candidate adoption, local-convergence or circuit claim. Targets are reliable
shared-input identification and eventual frozen native extraction validation.

Price: zero body forwards, same sample counts as the earlier pilot, plus one
analytic gradient and two analytic FD evaluations; batch64, FP64,300-second
managed hard timeout. Retain all native weight/background costs.
