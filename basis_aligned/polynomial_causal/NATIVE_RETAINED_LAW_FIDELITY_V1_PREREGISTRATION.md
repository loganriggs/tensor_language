# Native retained-law fidelity V1 preregistration

Frozen 2026-09-14 20:19 UTC, before this runner accesses model scores or
layer-17 states. This is a diagnostic of the synthetic Gaussian weighting used
for the setting-2 retained-producer objective. It performs no fit and cannot by
itself promote the sparse operator.

Use the already frozen 120-row `MINIMAX_FRESH_CACHE_V1` panel: 24 historical
anchors followed by 96 fresh-syntax rows. Recreate its three upstream
trajectories (native, child removed, remainder removed), capture the three raw
inputs to block 17 and the inherited layer-0 head-2 values, and form the exact
fourth additive corner. Decompose the normalized head17.2 interaction into the
same three retained producer contractions A1/A2/A3 used by
`NATIVE_RETAINED_ERROR_V1`. Contract the frozen sparse-minus-exact output-block
tensor against the actual compact MLP-only background, exact compact state,
and exact MLP17/final-RMS denominator. All nine branch Gram moments remain
joint; no source, output, or branch is independently resampled.

Predictions are fixed as follows.

- **A, instrument:** 360 body forwards and 120 rows; historical cached compact
  linear states and final states replay within `1e-4` relative error; the three
  producer branches replay their executor and the complete Gram sum replays
  the direct error energy within `1e-10`.
- **B, scalar fidelity:** the native-law total relative mixed-numerator error is
  within 20% multiplicatively of the frozen four-seed synthetic mean from
  `NATIVE_RETAINED_ERROR_V1_RESULT.json`.
- **C, interaction fidelity:** after dividing each Gram by its own complete
  joint energy, native and synthetic mean Grams have cosine at least 0.90 and
  all three off-diagonal signs agree.
- **D, stability:** for each of the five fixed consecutive 24-row groups, the
  relative mixed-numerator error lies within 35% of the full native-panel value
  and all three off-diagonal Gram signs agree with the full native panel.

If B--D pass, the Gaussian law is adequate for comparing this already frozen
candidate on this circuit panel, without establishing a general native law. If
any fail, reject synthetic-law fidelity and do not use it to select or refit a
setting-2 operator. Preserve the exact failure; do not rescue it with a new
threshold, row split, rank, support, or denominator.

Price: 360 body forwards over 120 already frozen prefixes, 360 extra MLP17
evaluations, twelve selected output coordinates, FP64 analysis, no learned
constants, and a 180-second managed-GPU bound. Native weights, upstream state
generation, all nonselected readers, MLP17, normalization, and the suffix stay
external. This changes the stable-identification evidence for the conditional
interaction operator; it is not a compression-only sweep, corpus OOD test,
selectivity result, or whole-model adoption claim.
