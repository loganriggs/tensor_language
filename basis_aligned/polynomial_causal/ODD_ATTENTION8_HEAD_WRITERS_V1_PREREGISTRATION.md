# Attention8 head writers feeding O current values

Capture block8 attention's tensor immediately before `c_proj`. For head h,
zero all other 128-coordinate slices and apply the matching native `c_proj`
columns; the nine writes sum to the full attention8 output up to native FP32
rounding. Construct paired donor-minus-recipient framing hybrids for the full
attention write and each head write. Keep the actual block9 residual native and
insert only the induced O current-value framing delta. Native plus ten arms on
48 rows costs 528 body forwards under 180 seconds.

- `pred_a`: nine head writes sum to full attention8 output within `1e-5`;
  native and full-attention target/work-jobs scores replay the block8-writer
  artifact within `1e-5`; outside framing stays exact, values are finite, and
  exactly 528 forwards execute.
- `pred_b`: native cue is positive for at least 10/12 pairs and full-attention
  target-effect RMS is at least `1e-5` in each template.
- `pred_c`: at least one common head has paired target-cue error versus full
  attention of at most 35% in both templates.
- `pred_d`: the sum of all nine separate head behavioral cue effects differs
  from the full-attention cue effect by at most 10% in each template.

Every head's four-control ratios and errors are reported; no head is selected
before execution. This tests stable cross-head grouping at the isolated value
interface, not whole-attention sufficiency, a semantic label, corpus OOD,
compression adoption, fitting, or quantization.

