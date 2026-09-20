# Subject-number effect: conditional six-MLP reduction

v643 causally freezes attention12–17 writes to their baseline values, recomputing
attention11, all later MLPs, residuals and normalizers. It predicts the frozen
five-port pre-L11 joint intervention on160 opened prompts with1.66–3.95% relative
effect error across12 cells. Minimum cosine is.99959; baseline replay is exact.
This is an intervention-response omission test, stronger than v642's small
attention attribution alone, but does not establish that those heads are useless.

v644 tests48 new matched-position prompts with six noun pairs disjoint from the
v642 subject sets. Rows were generated and hashed before native access. This is
not a claim that the nouns never occurred in training or any historical task.
Subject edits and final answer readouts have separate positions. Native accuracy
is100% in all eight cells; true effectRMS ranges.583–3.269 logits.

| New template | Singular error | Plural error |
| --- | --- | --- |
| Beside the attractor, the subject | .04164 | .03449 |
| Beyond the attractor, the subject | .02975 | .03436 |
| The subject beside the attractor | .08343 | .09842 |
| The subject beyond the attractor | .07633 | .07451 |

All registered relative-error<=.10 and cosine>=.99 gates pass. The postposed
plural/beside result is close to the error boundary. Both word orders still test
a narrow subject-number task, not broad syntactic generalization or unrelated
behavior preservation. No thresholds were relaxed and no coefficients fitted.

## Explicit computation and price

After attention11 and MLP11, freeze the six later attention writes a_l. At the
answer position the remaining dynamic suffix is a local rational-quadratic chain:

```
h_l = lambda0_l * x_(l-1) + lambda1_l * x0 + a_l
x_l = h_l + D_l[(L_l h_l) * (R_l h_l)] / (mean(h_l^2) + eps) + bias_l
```

Then apply final RMS, the two is/are readers and separate elementwise softcaps.
The same attention backgrounds serve both baseline and edited branches. All six
MLPs retain their native factors. Inputs are post-block11 state, initial embedding
state and six baseline attention-write vectors at the answer position: eight
1152-dimensional vector ports, or9216 input values per example. Their generation
remains native and is charged. The fixed program has six sets of L/R/D/bias and
two residual coefficients, plus two1152-dimensional readers:95,560,716 float values
(382,242,864 bytes at float32), and27,648 bilinear products per evaluation in
addition to dense projections. It is much too large to call a simple circuit.

[conditional_mlp_chain.py](conditional_mlp_chain.py) executes this program from
plain tensors with no model object. An independent float64 test compares it to the
unnormalized rational-quadratic formula and checks that omitting bias changes the
result. Native v645 is registered to compare its selected-logit margins to the
frozen native suffix at1e-4 absolute tolerance. This is not yet a saved standalone
weight package, and no storage-compression result is claimed.

v645 has now passed native replay: maximum selected-margin error3.8147e-6;
the registered effect and capability gates still pass. Measured parameter and port
counts agree with the figures above. Eight prefix and16 suffix calls on the now-opened
48rows took1.17seconds after model loading. [Native executor receipt](../bilinear_quotient/circuits/followups/subject_attention_freeze_v645_result.json).

An executed posthoc error-geometry analysis finds aligned effect recovery96.4–97.8%
for preposed templates but90.2–92.8% for postposed templates. In the latter cells,
94–99% of squared error lies parallel to the true effect. This suggests omitted
amplification, not a general constant correction: the scale depends on position.
No correction was fitted or installed. [Diagnostic receipt](SUBJECT_CHAIN_ERROR_GEOMETRY_2026-09-20.json).

## What this changes

We now have a causally tested conditional boundary for deeper folding: six local
bilinear maps rather than six dynamic attention-plus-MLP blocks. This supplies an
executable dense baseline for sparse shared computations. Its repeated residual
input slots and normalizers must stay explicit; naively expanding the unnormalized
chain would reach degree64, while the actual chain is not polynomial. HT or a DAG
should preserve intermediate contractions and exploit shared features, rather than
materialize that tensor. Compression must preserve the measured intervention effect
and ultimately remove costly background ports. The five-property circuit goal is
still incomplete: selective unrelated-reader tests, stronger reuse, simple learned
features and complete token-input closure remain missing.

Receipts: [v643](../bilinear_quotient/circuits/followups/subject_attention_freeze_v643_result.json),
[v644](../bilinear_quotient/circuits/followups/subject_attention_freeze_v644_result.json).
Frozen rows: [position stress](SUBJECT_POSITION_STRESS_V644_ROWS.json).
