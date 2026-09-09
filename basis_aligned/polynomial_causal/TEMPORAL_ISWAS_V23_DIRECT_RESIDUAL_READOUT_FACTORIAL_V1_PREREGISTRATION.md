# V23 direct residual-readout factorial v1

The valid A12--M17 reciprocal atlas found no dominant complete module. This test distinguishes a
direct identity-carried residual readout from an unlocalized aggregate of later-module responses.

On all 64 immutable v23 rows, reproduce four paths: native base, native donor, four-head-removed
donor, and the removed path plus the exact block-11 residual correction. Let `dy` be that correction
at block-11 output. Because each later block begins with `lambda0[layer] * x + lambda1[layer] * x0`,
the exact no-module identity contribution at final hidden state is

`direct = dy * product(lambda0[12:18])`.

Let `response = rescued_final - removed_final - direct`. This is an exact residual-state telescope;
it assigns every downstream nonlinear attention/MLP response to the correction term without naming
one module. Decode `removed+direct`, `removed+response`, and their exact sum through the checkpoint's
final RMSNorm, unembedding, and softcap. This measures direct sufficiency, response-only effect, and
their decoder interaction.

Predictions:

A. Hash authority, four hook paths, final-state closure, native rescued-logit replay, finiteness, and
   exact price (4 forwards / 256 sequence evaluations) pass.
B. Direct identity carry recovers at least 0.65 of the rescued residual effect with cosine >=0.90,
   direction >=0.90, positive projection in both halves, and P/C leak <=0.15.
C. The downstream response is a real signed correction: absolute target projection >=0.05 and is
   stable in sign across halves.
D. The nonlinear decoder interaction between direct and response has absolute signed projection at
   most 0.10.
E. The analytic carry coefficient is finite, nonzero, and exactly the product of checkpoint
   `lambda0` values for blocks 12--17.

A failure is invalid. A/B/E pass identifies a direct residual-readout candidate, not a standalone
token program: native earlier state construction and the final checkpoint decoder remain retained.
No fit, rank selection, v25 access, or post-outcome module selection occurs.

