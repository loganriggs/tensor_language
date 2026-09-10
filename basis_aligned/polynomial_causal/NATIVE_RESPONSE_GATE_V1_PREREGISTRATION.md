# Native use and context transfer of the fixed selected-reader response program

10 September2026. Original handoff/pilot, two unembedding views. The previous
CPU compiler proves local response/composition given context state; it does not
produce that state. Existing MLP17/v184/v185 and consumer dossiers are prior art.

Use the frozen K(2x1152), a(2), e, and readers V=(e, normalized U_runs-U_run).
For normalized native input u, s=e^T u and tau=K u-2*a*s. An edit delta e gives
selected response delta*(tau+2*a*s)+delta^2*a. The initial s stays native.
Compare native tau, a single frozen reference tau, and cyclic next-row tau.
Reference: first original GERUND_SHARED_READER A1 base row, chosen by existing
order, not outcomes. No averaging, fitting, rank/gain/direction/site selection.

Reused fresh A1/A2/G/C16 each. Capture base and donor at MLP17; edit only its
post-RMS input at the final semantic position by delta=e^T(u_donor-u_base).
Do not renormalize the edited input. Remaining input coordinates stay base.
Run one native body for the old reference, then base/donor/actual edited body
per panel:13forwards193 sequences, <=900seconds. No fitting/gradients.

For each context scheme, evaluate the two-reader formula. To test its effect
on all logits, install the fixed minimum-norm output change
V^T(VV^T)^(-1)*predicted_selected_response at the native final MLP output.
This writer preserves those two linear readers, not the other outputs or final
normalization. Compare with the actual native scalar-input edit using the real
final RMS/U/softcap. Also report exact-context projection separately, so loss
from omitted output directions is separated from context substitution error.

Predictions, scored without outcome filtering:

* A instrument:13/193 body counts,36 output hooks/body, all64 native pairs
  capable, native residual/readout bridges at existing bars. FP64 compiled
  versus direct finite difference rel<=1e-10. Native selected-response versus
  exact FP64 formula uses elementwise tolerance .01+1e-4*abs(expected), max
  scaled error<=1. Native full scalar-edit response norm must exceed1e-4 on
  every panel. V*writer identity error<=1e-10. Hashes checked before model load.
* B fixed-reference context reuse: A and selected-response relative error<=.10
  on BOTH A1/A2 using the single old reference tau. Individual panels reported.
* C within-frame lexical context reuse: A and selected-response relative
  error<=.10 on BOTH A1/A2 for cyclic next-row tau. No best donor choice.
* D full-effect prediction: A and fixed-reference writer prediction of centered
  full-vocabulary edit effect has relative error<=.10 on BOTH targets.
  This is conditional effect prediction, not target-circuit sufficiency.

Report G/C and native-versus-predicted correct-token CE changes, task-margin
changes, and context errors. G remains the failed agreement control from the
distributed experiment; this test cannot erase that result. New prediction
semantics on previously opened texts are not pristine data or pretraining-OOD.

All native model parameters and initial states remain required. Additional
program2306 coefficients plus fixed writer2304 coefficients are charged. The
reference gate costs2 more. Passing B would simplify only this context gate;
passing D would still be a background-conditioned local response program.
CPU continuation: paired intervals for context error versus projection error,
and explicit sign of the local native effect; no rank rescue if D fails.
