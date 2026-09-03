# Rung 569 preregistration: numbered-list index successor native gate

For prompt $x$, correct list label $a$, and the set $D$ of single-token plain digit answers, define

$$
m_D(x,a)=z_a(x)-\max_{b\in D,\ b\ne a}z_b(x).
$$

A standard endpoint cell passes when at least 75% of groups have $m_D>0$ and the 2.5th percentile of 2,000
group-bootstrap mean margins is positive.

1. Both endpoints of the two-line and three-line state-shift families must pass.
2. Both endpoints of the surface, middle-label, and repeated-label invariance families must pass. Under the R568
   correction, changing an earlier middle label is an invariance test because the claim reads the final visible label.
3. For each endpoint of the step-two conflict, the logit of `final visible label + 1` must exceed the logit of the
   arithmetic `+2` continuation in at least 75% of groups with positive bootstrap lower mean margin.

FIT opens first; this circuit's SELECT opens only if every FIT cell passes. FINAL_TEST/OOD remain closed. Failure stops
legacy L8H7/L8H3/MLP8--14 localization on R567. Maximum price for this circuit is 478 unique sequences in 15 forwards,
zero backwards and no fitted parameters.
