# Changed-city source edge through head8.2 into head9.8-O

Recompute native squared attention for block8 head2 as a tensor indexed by
query, source, and 128 value coordinates. At every framing destination, split
the head output into the single changed-city source and all other causal
sources, then apply the exact head2 columns of attention8 `c_proj`. Construct
paired donor-minus-recipient block9-residual hybrids for full head2, city-only,
and other-only writes. Feed only the induced current-value difference through
head9.8 O, retaining recipient queries, keys, inherited values and suffix.
Native plus three arms on 48 rows costs 192 body forwards under 180 seconds.

- `pred_a`: source sums replay captured preprojection head2 within `1e-5` and
  city+other partition within `1e-10`; native and full-head2 target/work-jobs
  scores replay the head-writer artifact within `1e-5`; outside framing is
  exact, values finite, and exactly 192 forwards execute.
- `pred_b`: native cue is positive for at least 10/12 pairs and full-head2
  target-effect RMS is at least `1e-5` in each template.
- `pred_c` (direct city-source edge): city-only cue error versus full head2 is
  at most 35%, while other-only error is at least 50%, in each template.
- `pred_d` (opposing contextual-source edge): other-only error is at most 35%,
  while city-only error is at least 50%, in each template.
- `pred_e`: separate city and other behavioral cue effects compose full head2
  within 10% in each template.

Four-control ratios are reported. This tests a concrete source→head8.2-write→
head9.8-O-value edge without fitting; it does not establish whole-head
sufficiency, corpus OOD, compression adoption, or quantization.

