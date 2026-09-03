# Rung 564 preregistration: increment native capability and rule separation

R564 asks whether bilin18 supports the R563 counterfactuals strongly enough to localize a circuit. It is not a search
over heads, MLPs, ranks, or subspaces.

For a prompt with registered answer token $a$, let $N$ be every single-token leading-space digit from 0 through 120
plus the registered leading-space number words from zero through twenty. Define the numeric-candidate margin

$$
m(x,a)=z_a(x)-\max_{b\in N,\ b\ne a}z_b(x).
$$

For each split, family, and base/donor endpoint, a capability cell passes when at least 75% of groups have $m>0$ and
the 2.5th percentile of 2,000 group-bootstrap mean margins is positive.

- Prediction A: every endpoint cell for digit +1, number-word +1, and cross-format +1 passes.
- Prediction B: every endpoint cell for the meaning-preserving +1 surface rewrite, repeated-number/copy control, and
  step-two control passes. This requires the model to distinguish the proposed +1 rule from generic number output.
- Prediction C: the coherent base endpoint of the broken-middle family passes the capability bar. In at least 65% of
  groups, breaking only the middle number lowers its registered-answer margin, and the bootstrap lower mean drop is
  positive.

FIT opens first. SELECT opens only if every FIT prediction passes, and the identical bars apply there. FINAL_TEST and
OOD remain closed. The literal maximum price is 960 unique sequences, 30 forwards at batch size 32, zero backwards,
and no fitted parameters. A failure stops localization on R563 rather than permitting post-hoc family removal.
