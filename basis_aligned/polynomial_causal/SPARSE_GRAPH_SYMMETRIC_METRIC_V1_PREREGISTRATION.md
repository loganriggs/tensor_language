# Rescore the frozen graph in the actual quartic coefficient metric

Reuse composed_quartic_contraction_v1, whose native diagonal/permutation and
sampling controls already passed. Every graph edge is a two-reader product:
physical readers H^(-1/2)Q columns, writer multiplied by sqrt(2) off diagonal.
Thus the existing fully symmetric four-linear contraction evaluates it exactly.

Select the old best graph solely by its recorded paired-coefficient fit. No
refitting, text or corpus probes. Use4096independent Gaussian quadruples and
4096Rademacher quadruples, seeds120547/120551, batch64. Evaluate full and
centered U norms of native target and residual; four independent identity-
covariance vectors give an unbiased estimator of symmetric tensor norm squared.
Capture is1-mean(residual2)/mean(target2), not necessarily a projection in this
new metric. Estimate ratio SE using residual2-ratio*target2 per paired sample.
Estimated errors are not rigorous confidence bounds.

- pred_a: native/graph diagonal and permutation identities<=1e-9, finite estimates.
- pred_b: centered symmetric capture>=1.25times the frozen paired capture9.08584%.
- pred_c: Gaussian/Rademacher centered capture differs by<=3combined estimated SE,
  and each capture estimate has absolute SE<=0.01.

Also report the exact paired full-U capture using retained edge energies and
the earlier exact full-U paired total, alongside centered paired capture.
An improvement is a metric effect, not newly fitted structure or native fidelity.
A miss only limits symmetrization as an explanation for this frozen graph.
No change to prior swap/removal failures. Zero body forwards/zero text sequences,
8192synthetic coefficient probes,300second alarm, no large artifact.
