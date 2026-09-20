# Joint prediction passes while weaker-component composition fails

The preregistered native test passes instrumentation and the folded program's
joint-effect gate, but fails conditional-increment prediction. This is not a
complete compositional circuit.

| Method | Joint cells passing | Attractor increments passing | Subject increments passing |
|---|---:|---:|---:|
| Full output quadratic |62/64|51/64|62/64|
| Separable output quadratic |62/64|29/64|61/64|
| Quadratic state, exact readout |64/64|51/64|62/64|
| Oracle that ignores attractor |63/64|0/64|64/64|

The oracle uses the actual subject-only effect and is not a predictive baseline.
It demonstrates the weakness of total-effect scoring when attractor effects are
only0.35–10.9%ofsubject effects. The conditional tests were added before joint
outcomes were opened. Full cross curvature matters: removing it drops attractor
increment passes from51to29, even though joint scores barely change.

Worst state-program errors are9.434%joint,25.546%attractor increment and
10.465%subject increment. Modal maxima are2.883%,6.821%and2.900%, respectively.
Original10%number/5%modal thresholds remain unchanged. All13failed attractor
increments are full-size edits in beside_subject; half-size mixed edits pass.

Native/double differences<=7.66e-6, prior axis outcomes replay exactly,
compiled readout replay<=4.98e-14 and full first/second derivatives<=1.51e-14.
The maximum numerical discrepancy is0.0358%ofitsprimitive budget, far below
the prediction failures. Actual counts match the preregistration:12prefix,
84double,68native suffix calls,12first+12second logical state JVPs,16native
gradients/32Hessian rows,16+32compiled checks and64exact readout checks.

The folded consumer stores69coefficients per context, versus24for the full
output quadratic and20separable, including constants. It predicts a two-input
function rather than isolated rays, but native source and state-derivative
generation remain required. The univariate two-square rewrite does not apply.

## Negative-result audit

For attractor increments, error decomposes exactly into the attractor-only
approximation error plus the error in the mixed inclusion-exclusion term:

    I(s,t)=Y(s,t)-Y(s,0)-Y(0,t).

Every standalone attractor approximation passes; worst number error2.861%.
The interaction-only error still fails the same13cases and reaches25.236%.
Thus substituting exact marginal predictions would not rescue composition.
The failure is not simply a bad attractor-only predictor hidden by subtraction.

Sign-parity analysis of the worst cell finds the largest residual component
even in subject amplitude and odd in attractor amplitude. This is compatible
with a leading s²*t term, but parity also includes higher powers and does not
identify an attention head or establish a cubic-only explanation.

## Next exact weight-folding target

At the first attention operation after the patch, block11, only two token states
depend on the two axes. Attention has no row softmax; its normalizations are
token/head local. Therefore the mixed attention-write term can occur only on
the causal edge from the earlier edited token to the later edited query.
This claim must be checked against native attention, including cache and RoPE.

For that edge, two QK products give a3x3polynomial core P_h in query/key
amplitudes per head. Keep query head norms, key head norms, value input RMS
and cached values explicit. The exact mixed observable factors as

    sum_h DeltaQ_h(u)^T P_h DeltaKV_oh(v).

Query features are shared across output readers; key/value features retain the
affine current value and constant cached value. The implementation uses300
coefficients per context for9heads/4readers, with native background/source and
reader generation still charged. It is a joint contracted operator, not an
independent matrix-SVD compression or a full suffix model.

The planted normalized-head reference replays full and mixed edge observables
within1.00e-15, and zeroing either source direction kills the mixed term exactly.
Native localization, downstream causal relevance and prediction repair remain
untested. A baseline reader is a fixed linear observable; it is not proof of
the full finite downstream effect.

Receipts: `two_site_composition_v1r1_result.json` in followups;
`TWO_SITE_FAILURE_DECOMPOSITION_V1.json`, `ATTENTION_MIXED_EDGE_CONTROL.json`
and their reproducible audit/control scripts here.
