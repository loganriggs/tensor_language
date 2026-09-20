# Upstream closure of the output component

Follow-up: [the complete recent-path executor](RECENT_FOLDED_COMPONENT_2026-09-20.md)
recomputes attention17 and reduces MLP16 intervention prediction errors to
2.1–3.1%, while explicitly charging the remaining dense local weights.

The previous256-square program predicts a selected quadratic component from
the actual normalized MLP17 input. That conditional extraction leaves its
upstream feature generators unexplained. These experiments test that gap.

## Exact residual-source expansion

Carry every native write through the actual residual lambdas to the MLP17
pre-normalization state h. Include the initial normalized embedding and each
subsequent embedding injection once,18 attention writes, and17 earlier MLP
writes:36 sources in total. MLP output biases stay inside their producing
source. The target MLP17 bias is outside the quadratic being predicted.

For h=sum_s h_s and Q=P diag(c) P^T, retain the entire pair table

M_st = (P^T h_s)^T diag(c)(P^T h_t)/(mean(h²)+eps).

The sum over ordered pairs is the quadratic prediction. Off-diagonal pairs
appear twice; row sums allocate half each cross interaction to its endpoints.
These are fixed-denominator attributions, not causal effects. Exact fixed-norm
leaveout is2*row_sum_s-M_ss; actual normalized leaveout also changes the
denominator. Tests verify the residual recurrence, pair accounting, and greedy
norm-closed selection against explicit subset evaluations.

v625 native reconstruction error is1.59e-7; pair-table replay error7.54e-7.
The cross-term sum has L2 norm .774/.768 times the complete program's norm on
the two panels. Self and cross terms are not orthogonal, so these are not
additive explained-variance shares.

## Attribution ranking does not find a compact input circuit

v625 ranks mean absolute row attribution on skip1200 and freezes that order
for skip11000. The first eight sources are MLP16,MLP15,attention5,MLP11,MLP12,
MLP13,attention6,attention7. Recompute RMSNorm from the retained sum; do not
quietly retain the original denominator as an oracle.

v626 instead greedily removes the source producing the smallest actual
calibration squared error after recomputing the denominator. It freezes its
support order before the second panel. It is a heuristic, not a global optimum.

| Retained sources | Attribution order: calibration / validation | Norm-closed greedy: calibration / validation |
| --- | --- | --- |
| 8 | .87149 / .93494 | .30365 / .32050 |
| 16 | .17592 / .17067 | .17592 / .17067 |
| 24 | .06224 / .06086 | .06035 / .06261 |
| 36 | .04592 / .04165 | .04592 / .04165 |

Errors are relative to the exact native quadratic component, excluding its
bias. The full36-source row retains the256-square approximation error. Both
methods fail the registered8-source<=.10 gate.24-source subsets pass that
error level, but each retained write still needs its native upstream generator.
This neither proves that all8-source subsets fail nor extracts a smaller
end-to-end circuit. These panels were previously opened for other tests and
are not new domain-OOD evidence.

## Causal closure test

v627 compares two different operations for MLP16 and attention5, nominated
from the calibration census:

1. Subtract only the carried baseline source at h17, keep all other writes
   frozen, then recompute RMSNorm and the exact native MLP17.
2. Zero that upstream module's full output in the actual model and let every
   downstream operation recompute before reading the target component.

Using exact native MLP17 on both sides removes the256-square approximation
as an explanation for disagreement. The receipt reports relative errors of
predicted *changes*, the true effect size, and downstream state drift. Passing
baseline replay is necessary before interpreting those results. A whole-write
ablation is a causal boundary intervention, not a semantic-selectivity test.

v627 finishes with baseline replay5.11e-8:

| Source | Effect-prediction error: calibration / validation | Downstream state drift: calibration / validation |
| --- | --- | --- |
| MLP16 | .09256 / .07550 | .10951 / .10817 |
| attention5 | 1.31612 / 1.33240 | .79085 / .78467 |

The MLP16 prediction passes the registered<=.10 gate; attention5 fails.
Both interventions have large effects on the component, so this is not a
vanishing-effect denominator artifact: MLP16 change is1.26/1.36 times the
baseline component L2, attention5 change .77/.79. Whole-write CE damage is
.955/1.161 for MLP16 and2.110/2.289 for attention5, which gives no evidence of
selectivity by itself.

The next folding target is the recent MLP16-to-MLP17 branch. Pull the selected
MLP17 input features through Down16 into its bilinear products, retain the
background/cross terms and actual RMSNorm denominator, and explicitly account
for attention17 recomputation (the remaining source drift). This is where the
shared hierarchical representation becomes useful. Attention5 needs a broader
downstream generator model; it cannot be treated as a fixed additive source
under intervention. Neither result establishes full upstream extraction.

The authoritative receipts are `output_component_sources_v625_result.json`,
`output_component_sources_v626_result.json`, and
`output_source_causal_closure_v627_result.json` under
`../bilinear_quotient/circuits/followups/`. Scripts run through the managed
queue; v625/v626 each use4 model forwards, v627 uses12 plus8 local MLP calls.
The full OOD/extraction/removal/reuse/simplicity goal remains open.
