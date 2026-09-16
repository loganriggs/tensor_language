# Equality L5H5 M4 moment-weighted CP-rank certificate V1

The isotropic reader-weighted tensor has a strong CP-rank obstruction. This
follow-up tests the DCT briefing's proposed escape hatch: contract the input
slots with empirical activation moments rather than the identity metric, and
keep natural and code contexts separate rather than averaging them together.

Using the already frozen 192 natural and 192 code documents, collect the second
moment of the normalized MLP4 input on each panel. For each moment, whiten the
two input slots of the exact rank-256 L5-reader tensor and compute both input-
mode unfolding spectra. Also compare the Gaussian/Isserlis prediction for total
same-state quadratic energy with the directly observed energy on that panel.

Predictions fixed before execution:

1. RMS-normalized-state and spectral instruments pass: moments are PSD within
   `2e-4`, mean state-square lies in `[.98,1.02]`, and the two transformed tensor
   energies agree within `2e-4` on each panel.
2. Natural activation moments reduce the rank-256 CP error lower bound by at
   least 20% relative to the isotropic authority (`.7588316302`).
3. The natural rank-256 lower bound is at most `.50`.
4. The frozen code rank-256 lower bound differs from natural by at most `.10`.
5. Gaussian/Isserlis total same-state energy differs from directly observed
   energy by at most 25% on each panel.

Failure of prediction 1 is invalid. Other failures are valid metric/context
nulls. Passing 2--5 is only evidence that a moment-weighted CP fit is worth
attempting; no extraction, removal, behavior, or product-cost claim follows.

Price: one checkpoint load; 192 natural and 192 code partial-prefix executions;
two 1,152-dimensional moments and four unfolding spectra; no gradients, fits,
parameter updates, or new text.
