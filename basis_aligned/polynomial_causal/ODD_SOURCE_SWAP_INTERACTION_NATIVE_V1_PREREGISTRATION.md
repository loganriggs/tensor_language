# Routing, value, and mixed terms in O framing-source interchange

Reuse the frozen paired framing swap. With the recipient final query fixed,
factor each framing-source O contribution as odd routing `r` times mixed value
`v`. Donor source replacement obeys
`r_d v_d-r_o v_o=(r_d-r_o)v_o+r_o(v_d-v_o)+(r_d-r_o)(v_d-v_o)`.
After 48 native captures, run full, routing-only, value-only, and additive
routing+value swaps. The last omits the mixed term. Total price is 240 body
forwards under 180 seconds.

- `pred_a`: the generalized source kernel replays the frozen helper and the
  three-term source expansion within `1e-10`; native and full-swap target plus
  work/jobs scores replay prior artifacts within `1e-5`; the mixed source term
  is live and all 240 results are finite.
- `pred_b`: native target cue is positive for at least 10/12 pairs and full-swap
  target-effect RMS is at least `1e-5` in each template.
- `pred_c`: additive-no-mixed paired target-swap effect differs from full swap
  by at most 10% in each template.
- `pred_d` (value hypothesis): value-only error versus full swap is at most 35%
  and routing-only error at least 50% in each template.
- `pred_e` (routing hypothesis): routing-only error is at most 35% and
  value-only error at least 50% in each template.

Four unrelated control ratios are reported for every arm. A C failure makes the
routing×value interaction behaviorally necessary at this boundary; a pass only
permits its omission on this panel. Native upstream states and suffix remain.
There is no fitting, upstream necessity, corpus OOD, compression adoption, or
quantization claim.

