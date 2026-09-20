# Gaussian versus empirical calibration moment contractions at fixed graph cost

2026-09-20 22:10 UTC. Matching actual mean/covariance still predicts positive
mode1 correction alignment where measured alignment is negative. The mismatch
is principally in residual/correction alignment (moments through degree6),
not the correction's fourth-moment energy. Do not repeat input-covariance sweeps.

This comparison explicitly extends beyond an input covariance: use empirical
function moments from ORIGINAL calibration inputs and weight-evaluated native
polynomial targets, already archived in NATIVE_VARIATION_AUDIT_V1.pt panel0.
It is data-informed polynomial regression, not a weights-only objective, and
uses neither task labels nor outputs from any diagnostic/test panel.

Freeze base ten-product graph, its six products p, Gaussian center c and
output rank2. q=p-c. Gaussian moments K0=E[(F-base)q^T], G0=E[qq^T] are archived.
Compute calibration K1,G1 without subtracting empirical q means: centering is
fixed at c. Blend K_alpha=(1-alpha)K0+alpha K1 and similarly G_alpha, with
alpha in{0,.25,.5,.75,1}. Solve the same rank2 least-squares readout analytically.
All arms cost21948coefficients/10products. Every correction retains zero mean
under the ORIGINAL calibration Gaussian because c stays fixed.

Primaryalpha=.5 selected before outcomes. Other arms diagnose sensitivity;
no later alpha selection may be presented as preregistered confirmation.
- pred_a_replay: alpha0 functional replay of archived quadratic rank2<1e-5,
  all G positive definite and all solves finite; target/input token hashes and
  teacher scale agree.
- pred_b_calibration: alpha1 empirical calibration MSE <= alpha0, and alpha0
  Gaussian objective <= alpha1 (same graph class, exact endpoint objectives).
- pred_c_transfer: primaryalpha.5 code mode1 native-removal error<.40 and
  FineWeb branch CE<.025. All alpha outcomes and Gaussian objective costs kept.

Diagnostic panels remain reused. A successful arm must subsequently be frozen
and tested on untouched contexts/corpora and selective edits; no circuit
promotion or assumption of monosemanticity follows from this test.
