# Upstream layer onset of the O framing current-value cue

Use the 48 frozen fresh rows. A native pass captures every block's residual
input. For each layer `L=0..9`, donor-patch framing residual positions at block
L and propagate through block8. At block9, capture the resulting normalized
attention input, restore the entire recipient block9 residual, and insert only
the induced O current-value change at framing sources with recipient queries,
keys, inherited values, other heads, residual path and suffix. This is 48 native
plus 10x48 interventions: 528 body forwards under 180 seconds.

- `pred_a`: layer0 target effect is at most `1e-8`; native and layer9
  target/work-jobs scores replay the frozen current-value artifact within
  `1e-5`; restored block9 attention inputs and outside-mask patches are exact;
  all outputs are finite and exactly 528 forwards execute.
- `pred_b`: native cue is positive for at least 10/12 pairs and layer9 target
  effect RMS is at least `1e-5` in each template.
- `pred_c` (early onset): in each template at least one layer at most 4 has
  target-effect norm at least 50% of layer9 and cosine at least `.8` to layer9.
- `pred_d` (opposing late onset): in each template every layer 0..6 has norm
  below 20% of layer9, while layer8 has norm at least 80% and cosine at least
  `.8`.

The full per-layer profile and four-control ratios are reported. This isolates
the arrival of information at the identified O current-value interface; it is
not a generic residual patch, unique semantic unit, corpus OOD, compression
adoption, or quantization claim.

