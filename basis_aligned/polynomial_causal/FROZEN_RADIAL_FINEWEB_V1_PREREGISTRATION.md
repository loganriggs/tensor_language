# Frozen radial-corrected weight programs on FineWeb

Validate three already frozen weight-derived dictionaries: orthogonal seeds 0
and 937, and ordinary-covariance oblique seed 0, each with/without its exact
radial correction. Include bias-only and radial-plus-bias-only baselines: eight
candidate arms. No fitting, support selection, hyperparameter selection or
candidate selection using text. Uniform-sphere capture is a hypothesis source,
not an assumption that FineWeb states are uniform.

Use the first 64 rows and first 129 tokens of the existing
`fineweb_n192_skip7000.pt` cache: 128 shifted prediction positions per row,
8192 positions total. This cache has been opened by prior work. It is FineWeb
training-corpus validation, not fresh/document-held-out or OOD evidence.
Freeze cache and row hashes before enqueue. Pile is not used.

Run the native prefix once per eight-row batch and capture the exact final-MLP
normalized input, residual before that MLP, and native MLP output. Substitute
each frozen program at that interface, preserving native final RMS, unembedding
and tanh cap. Score in 256-position chunks. Save interfaces and per-position
scores for validation-only followups, not all vocabulary logits.

On the first eight rows, compare this path against two physical full forwards:
native, and a final-MLP hook replacing its output with corrected orthogonal0.
The hook must see exactly the captured input. Check complete logits and native
mean cross-entropy. Total ten body forwards / 80 sequence evaluations of 128
tokens; the eight candidate readouts do not rerun the prefix. Explicit body
counting at the first attention module enforces this price.

Report full-U-weighted MLP squared error divided by native MLP output energy,
KL(native distribution || candidate distribution), cross-entropy and CE added
above native (positive is damage). Report all 64 paired row means; aggregate
MLP error is a ratio of summed energies. Candidate computation uses saved FP64
programs and casts outputs to the native FP32 interface before readout. Bias,
residual, input radius, final RMS and tanh are preserved explicitly.

Predictions, frozen before any text scores:

- A: both physical-control mean CE errors <=1e-5; full-logit relative L2 errors
  <=1e-6; hook-input relative error <=1e-6; body counts exactly 10/80; all scores
  finite. Failed replay invalidates downstream behavioral interpretation.
- B: each of three radial corrections reduces aggregate U-weighted MLP squared
  error by at least 10% relative to its uncorrected partner.
- C: each of three corrections reduces mean KL by at least 10% relative.
- D: radial-plus-bias-only has lower mean KL than bias-only.

A correction can improve weight/sphere error while worsening FineWeb behavior;
that is the opposing prediction. Record B/C/D separately and preserve failures.
No threshold is a circuit-identification or adoption gate. CE improvement alone
does not establish selective removal, extraction, OOD prediction or reuse.

Native background and U remain charged. Each complete sparse reader program
contains 7,815,168 matrix coefficients plus 1,179,648 support indices and 1152
bias values; the correction adds 1152 numbers. Native Down is retained. This
test neither measures a runtime speedup nor claims independent circuit export.
Managed lane1, appended after existing weight-only discovery, 900-second alarm.
