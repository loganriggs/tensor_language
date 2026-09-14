# Block8 writer split feeding O current values

Capture paired block8 residual input, x0 re-entry, attention write, MLP write,
and block9 residual. Their exact donor-minus-recipient identity is
`delta r9=lambda0*delta r8+lambda1*delta x0+delta attn8+delta mlp8`.
At framing positions construct full, input-carry-only, attention8-only, and
MLP8-only hybrid block9 residuals. Keep the actual block9 residual native and
insert only each hybrid's induced O current-value framing delta, with recipient
queries, keys, inherited values and suffix. Native plus four arms on 48 rows is
240 body forwards under 180 seconds.

- `pred_a`: the residual identity holds within `1e-10`; paired framing x0 delta
  is exactly zero; native and full-hybrid target/work-jobs scores replay the
  frozen current-value artifact within `1e-5`; outside-mask changes are zero,
  values finite, and exactly 240 forwards execute.
- `pred_b`: native cue is positive for at least 10/12 pairs and full-hybrid
  target-effect RMS is at least `1e-5` in each template.
- `pred_c`: attention8-only cue error versus full is at most 35%, while carry
  and MLP8 errors are each at least 50%, in each template.
- `pred_d`: MLP8-only error is at most 35%, while attention and carry errors are
  each at least 50%, in each template.
- `pred_e`: carry-only error is at most 35%, while attention and MLP errors are
  each at least 50%, in each template.
- `pred_f`: the sum of the three separate paired behavioral effects differs
  from full by at most 10% in each template.

Controls are reported. These are exact path hybrids at the already isolated O
current-value interface, not whole-block behavioral interchange, corpus OOD,
compression adoption, fitting, or quantization.

