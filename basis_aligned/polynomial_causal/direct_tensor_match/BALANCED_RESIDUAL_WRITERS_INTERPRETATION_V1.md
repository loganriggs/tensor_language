# Smaller-output residuals do not collapse into four shared effects

22 September 2026, 05:25 UTC. Both frozen CP512 parents, outputs4–15. Calibration uses6144 states; evaluation uses16384 opened states and2494 fixed matched-state differences. Scale each output by its calibration target RMS before comparing residual directions. This differs from the earlier unbalanced all-output PCA and from the capped Gaussian training weights.

| Measurement | Seed1001 | Seed1002 |
| --- | ---: | ---: |
| Four calibration directions: evaluation value capture |56.18%|57.56%|
| Same four directions: response capture |57.78%|57.63%|
| Eight calibration directions: value capture |87.84%|86.92%|
| Same eight directions: response capture |88.80%|88.21%|
| Best evaluation-specific four directions: value capture |60.66%|60.22%|
| Best evaluation-specific four directions: response capture |60.63%|59.55%|

The registered prediction of at least90% rank-four capture on both metrics for both starts **fails**. The evaluation-specific optima show that this is not merely a bad calibration basis: residual variation across smaller outputs remains broad. All finite-value, token-alignment, planted rank-two, orthogonality, energy-accounting and full-rank closure checks pass.

Let A be the evaluation residual matrix after fixed calibration scaling. An additive correction using at most four fixed output writers has the form H W transpose and therefore matrix rank at most four on this panel, even if H could contain arbitrary scalar functions of the input. Its best possible squared residual obeys

$$
\min_{\operatorname{rank}(C)\le4}\|A-C\|_F^2
=\sum_{j>4}\sigma_j(A)^2.
$$

Thus even oracle scalar amplitudes with four optimally chosen writers leave roughly63% of the current residual RMS. This percentage is relative to the current residual, not the native target. The value and response optima use different oracle bases; neither is a trained executable correction. A free uncharged offset would change the allowed class and is not included in this bound.

This supports retaining broader output ports in the queued output-local learner. It does not justify twelve separate semantic labels, exclude shared intermediates feeding many writers, prove population/OOD behavior or establish minimal arithmetic complexity. It also does not negate the earlier observation that a few directions explain most *unbalanced all-output* error: that metric lets large components dominate. The two measurements ask different questions.

No candidate, basis or corrective amplitudes were exported. No native job changed. The next discriminating actions remain native new-feature fitting and removal-stage geometry, rather than another output-readout sweep.

[Script](audit_balanced_residual_writers.py) · [Registered prediction](BALANCED_RESIDUAL_WRITERS_PLAN_V1.md) · [Full receipt](BALANCED_RESIDUAL_WRITERS_V1.json).
