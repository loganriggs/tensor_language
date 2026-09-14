# Source positions of the odd key-parity branch

Use all48 article-corrected rows and derive source masks only from paired token
differences. At head9.8, partition the complete reflection-odd branch `O` into
attention contributions from the changed city positions and every other source
position. Query state, keys, values, native normalization, rotary positions and
the complete suffix remain live. No weights, rows, ranks or thresholds are fit.

Four arms are native, remove `O_city`, remove `O_other`, and remove `O_all`.
There are192 body forwards with a180-second cap.

- `pred_a`: native and all-O-removal outputs replay the corresponding frozen
  article-correction cube arms within `1e-5`; live `O_city+O_other=O` error is
  at most `1e-10`.
- `pred_b`: the city-only removal cue-effect vector approximates all-O removal
  within20% relative L2 in each family.
- `pred_c`: all-O cue-effect norm is at least5% of full-head removal cue norm,
  and city-only control-effect RMS is at most50% of city-only target-cue RMS in
  each family. Denominator floors are `1e-8`.

The null is that O's cue effect is distributed through context-dependent sources
or fails selective materiality. `REGIONAL_SOURCE_POSITIONS_V1` tested a different
compiled mixed-token component and failed city localization;
`INHERITED_SOURCE_POSITIONS_NATIVE_V1` tested only inherited even values. This
screen concerns O. It does not label O semantically, establish OOD behavior,
remove upstream city descendants, compress static weights or use quantization.

