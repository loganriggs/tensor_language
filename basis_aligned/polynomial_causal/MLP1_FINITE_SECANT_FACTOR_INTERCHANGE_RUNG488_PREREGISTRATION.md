# Rung 488 preregistration — BF16-calibrated MLP1 finite-secant factor interchange

## Decision

Rung487's exact float32 polarization identity held at relative squared error `1.98e-12`, all calls and replays were
exact, and the deployed end-to-end branch identity held at `5.48e-6`. Its instrument nevertheless failed because
two separately registered BF16 relative-squared tolerances were both `1e-5`: direct BF16 polarization measured
`7.61e-5`, and adding the OWN secant to the branch-absent MLP1 write measured `3.62e-5`. Validation therefore stayed
closed. The T--I shared-live-state result is descriptive only and rung487 remains a failed instrument.

Rung488 is the named b-variant. It reruns the same discovery intervention and opens held-out documents only if the
repaired instrument and every original discovery gate pass. It changes no branch, factor, arm, edge, effect, or
held-out threshold.

## Precision repair

For round-to-nearest BF16, unit roundoff is `u=2^-8`. A relative-squared comparison naturally scales as `u^2` rather
than `u`. Freeze these deployed-dtype bars before rerunning the model:

- direct BF16 polarization identity: at most `8u^2 = 1.220703125e-4`;
- `branch-absent MLP1 write + OWN secant` versus native MLP1 write: at most
  `4u^2 = 6.103515625e-5`.

The multipliers allow several rounded bilinear operations and cancellation while remaining under `0.0123%` and
`0.0062%` relative squared error. The original float32 polarization bar (`1e-8`), deployed complete-branch identity
bar (`1e-5`), exact prefix replay, exact call counts, exact injection counts, and nonzero-effect checks are unchanged.
The same repaired bounds apply independently to discovery and validation; validation does not inherit discovery's
measured errors.

## Computation and frozen scientific predictions

Use exactly rung487's 1,000 hash-bound documents, T/C/I branch definitions, MLP1 secant formula, six ordered branch
pairs, OWN/CONTEXT/DIRECTION/BOTH arms, 16 shifted-position controls, and physical suffix recomputation. Discovery is
documents0:500 with halves0:250 and250:500. Validation is documents500:1000 with quarters500:750 and750:1000.

- **A:** all unchanged exactness checks pass plus the two repaired BF16 bounds above.
- **B:** every OWN effect has RMS at least `.10 nat` in both discovery halves and both half-scale ratios lie in
  `[.80,1.25]`.
- **C/D:** discovery independently recovers exactly one bidirectional T--I CONTEXT edge. In each direction and half,
  physical-effect cosine is at least `.80`, best scalar-adjusted relative error is at most `.50`, CONTEXT beats
  DIRECTION by at least `.15`, and same-position MLP1-write cosine beats the 95th percentile of 16 shifted controls
  by at least `.15`. T--C and C--I must remain absent; extra edges fail the frozen graph prediction.
- **E:** only if A/B/D pass, freeze that graph and open validation. The T--I CONTEXT edge must pass every original
  clause in both validation quarters, with the repaired instrument independently valid there. No edge may be added.

The strong null fires if A, B, or D fails. A descriptive discovery edge is not a claim if E fails. If E holds, the
result identifies a shared continuous MLP1 live-state factor for T and I: their branch-induced changes differ, but
swapping the live state that multiplies those changes preserves their downstream causal effects.

## Relevance and price

This tests cross-module grouping and held-out causal interchange, not rank reduction. It addresses stable
identification, held-out prediction, and selective factor swapping. It does not by itself establish a complete MLP0
explanation or an adopted compressed model.

Price is unchanged: 3,500 full-model discovery forwards at batch size4 and another 3,500 only after the discovery
license. No deployed parameters are added or removed; store only contracted statistics and hashes.
