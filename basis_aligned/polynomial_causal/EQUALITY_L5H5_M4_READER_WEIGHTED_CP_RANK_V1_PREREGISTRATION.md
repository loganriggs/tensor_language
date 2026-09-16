# Equality L5H5 M4 reader-weighted CP-rank certificate V1

The normalized-input executor still evaluates all 4,608 native MLP4 products.
This weight-only experiment asks whether the retained rank-256 L5H5-reader
tensor can be replaced by a small asymmetric CP node

`sum_j output_j * (left_j @ x) * (right_j @ x)`.

Construct the exact retained tensor from MLP4 Left/Right weights and the
singular-value-weighted 256-dimensional L5 Q/K reader contraction. Compute both
1,152-dimensional input-mode Gram matrices without materializing the full
third-order tensor. The best rank-`r` matrix approximation of either unfolding
is a lower bound on the error of every CP-rank-`r` executor in this metric.

Predictions, fixed before execution:

1. Four random contractions agree with each implicit Gram quadratic form within
   `5e-4` relative error; the two total tensor energies agree within `2e-4` and
   numerical PSD error is at most `2e-4` of the largest eigenvalue.
2. Every CP-rank-64 executor has at least `.50` relative reader-weighted tensor
   error.
3. Every CP-rank-256 executor has at least `.25` relative reader-weighted tensor
   error.
4. Every CP-rank-512 executor has at least `.10` relative reader-weighted tensor
   error.
5. The retained 256 output coordinates are numerically orthogonal after native
   reader weighting: maximum normalized off-diagonal Gram entry at most `2e-4`.

Failure of prediction 1 or 5 is invalid. Failure of 2--4 is positive evidence
for the corresponding compression scale, not an instrument failure. Passing is
an obstruction only in the declared reader-weighted Frobenius/independent-
Gaussian metric; it does not rule out context-gated, activation-moment-weighted,
or behavior-specific compression.

Price: one checkpoint load, one existing rank-256 basis compilation, two
4,608-channel Gram contractions, no prompts, model forwards, gradients, fits,
or parameter updates.
