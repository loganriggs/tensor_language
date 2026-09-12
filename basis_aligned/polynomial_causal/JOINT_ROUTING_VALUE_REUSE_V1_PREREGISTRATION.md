# Joint routing/value shared-function screen

Discover from weights only. Evaluate raw attention17 q2/s3 coefficient features
in each head's128 value-output coordinates, before O. Allow different output
writers: compare functions via uncentered coefficient CCA, not OV alignment.
All head normalization gates remain private downstream factors.

Training:4096 independent multilinear Rademacher probes,2048 each at query8 /
source0 and7, seed73300. Fit CCA for all36 head pairs, spectral whitening cutoff
1e-8 of maximum eigenvalue, rank8. Select the pair with largest mean of its
eight canonical correlations; tie by lexicographic head pair. Freeze readers.
Validation:4096 new coefficient probes with the same position balance,
seed73301. Separately2048 diagonal normalized query/source ports per position,
seed73302. No text, task labels, output behavior or native activations for fitting.

- A: planted shared-function and joint-polynomial controls hold; native diagonal
  routing/value replay<=1e-10; whitening identity<=1e-8; runtime<=180seconds.
- B: at least4 of8 fixed modes have training correlation>=0.95 and independent
  coefficient cosine>=0.90 at BOTH source positions.
- C: at least4 of those same modes also have <=0.10 relative RMS error in BOTH
  directions and BOTH positions when a shared numerator predicts the other
  head's scalar computation with that destination's native gate restored.

Null: no substantial shared eight-mode pair is resolved by this screen. This
does not exclude nonlinear recoding, a smaller shared component, multiple-head
combinations, different positions or output-conditioned reuse. A selected-pair
failure must retain all36 training scores and independent-mode diagnostics.
No OOD text, extraction, selective intervention or circuit promotion from CCA.

Price: zero body forwards, one managed GPU, batch64 FP64,240-second hard limit.
No large coefficient tensor or training feature bank is saved. Save covariance,
readers and selected validation projections for audits. Any eventual implementation
must price both readers/writers, all native QK/value factors and private gates;
correlated outputs alone do not save computation.
