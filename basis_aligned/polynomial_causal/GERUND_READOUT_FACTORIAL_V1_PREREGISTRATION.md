# Grammatical MLP17 response: token numerator versus final RMS

10 September2026. Existing MLP17 calibration/two-consumer work is prior art.
This tests the missing output effect of the fixed grammatical scalar-input edit,
not whether normalization matters in general. No new directions, gates or ranks.

Reused fresh A1/A2/G/C16, native base/donor/edited body each:12forwards192seq.
Same post-RMS MLP17 input edit delta=e^T(u_d-u_b) as NATIVE_RESPONSE_GATE_V1.
Replay its native CE/recovery and exact two-reader projection errors. Save
base/edited final states, base MLP input and folded coefficients for CPU reuse.

Write z(n,d)=30*tanh((U h_n)/(30*RMS(h_d))). Evaluate native00, full11,
numerator-only10 and denominator-only01. Their endpoint interaction is
z11-z10-z01+z00; its norm is descriptive, not additive causal shares.
Also evaluate the old exact two-reader projected state with its own denominator,
then with the true edited denominator. This tests whether supplying the missing
normalization alone suffices to repair its full-vocabulary effect prediction.

For the same local input path, h(delta)=h0+delta*b+delta^2*a, where
b=D[(Le)*(Ru)+(Lu)*(Re)] and a=D[(Le)*(Re)]. Fold actual answer and foil token
rows of U into the three numerator coefficients. Final RMS squared is the
degree4 polynomial with coefficients [h0^2,2h0.b,2h0.a+b^2,2b.a,a^2]/1152,
plus native float32 epsilon in the constant. Two token numerators and one
shared norm polynomial need11 numbers per initialized context. Shift these
polynomial coefficients for sequential commands; no fitted approximation.
This is a conditional token-score program, not initial-state production or
full-vocabulary CE prediction. It needs native h0/u and weights to initialize.

Predictions:

* A instrument:12/192, all64 native endpoint pairs capable, existing state
  bridges; factorial00/11 vs native logits maxabs<=1e-3/rel<=1e-5. Prior CE,
  recovery and exact-projection full-error replay maxabs<=1e-3. FP64 tiny
  moment/composition errors<=1e-10, no-op nonlinearity semantics unchanged.
* B norm-only explains effect: A and centered full-vocab effect error<=.10
  for denominator-only01 on BOTH A1/A2.
* C numerator-only explains effect: A and the same bound for10 on BOTH targets.
* D true norm repairs two-reader writer: A and the same bound for projected
  numerator plus true edited denominator on BOTH targets.
* E token/moment program: A; answer/foil score maxabs error<=1e-3 on all64 rows,
  task-margin maxabs error<=1e-3, and half-command composition error<=1e-10
  versus evaluating the full command in FP64. No full-model extraction claim.

All controls and signed CE changes reported. Runtime<=900managedseconds. All
545902902 native parameters remain dependencies; initialized token program has
11 numbers/context and saved endpoint state costs are counted. No tensor-rank
or prototype rescue. CPU continuation: paired effect-error/interaction intervals
and saved-state audit, separating token-numerator information from norm effects.
