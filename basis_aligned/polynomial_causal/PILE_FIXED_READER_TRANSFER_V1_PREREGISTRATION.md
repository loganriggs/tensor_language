# Fixed input-product transfer to Pile, 10 September 21:25 UTC

Use the captured million-token panel to ask whether already learned input
computations transfer across corpus and position coverage. This is a functional
transfer screen, not an OOD circuit certificate. Prior MLP17 affine/calibration
baselines and current fit failures checked; claim on board.

Freeze five functions before evaluation: first data product128, data shared64/
product128, data block32x8, original weight product128, penalized weight product128.
Source files and this registration are hash-bound. Evaluate each unchanged
function on Pile validation. Then hold all input readers/products fixed and
refit only output writers on Pile training. Compare unchanged and writer-refit
functions on the same validation inputs. Test split remains unopened.

Training/validation:51200/7168 sampled inputs,1600/224 distinct document prefixes.
All sampled positions0..511 form primary full-position tables. Also report
positions>=64 separately because the original fitting panel excluded earlier
positions. Readers and refitted writers are never selected using validation.
Regenerate bias-free native bilinear targets in FP64 on decoded cached inputs.
Capture replay already validates this input-rounding approximation; wholemodel
finalRMS/tanh and behavioral effects remain outside this measurement.

Full-U squared error via M=U^T U and Cholesky root H: target Z=YH. For each fixed
feature matrix F, solve F beta≈Z by untruncated QR. Then W=H^{-T} beta^T. This is
the exact conditional objective, no new penalty/ridge. Compare QR with full SVD
and normal equations on training features, using function outputs rather than
unstable individual coefficient differences. Constant and affine baselines fit
on training; affine uses normal equations only if its Gram condition<=1e12.

Registered predictions:

- A instrument: five full-rank feature solves; QR/SVD function relativeL2<=1e-8,
  QR/normal-equation function relativeL2<=1e-6; native dimensions/counts finite;
  writer-only refit cannot increase training squared error by>1e-8. Toy objective
  and first-gradient controls must pass before enqueue. Failed numerical bridge
  is an invalid solver comparison, not structural absence.
- B reusable shared inputs: unchanged shared-reader validation error on matched
  positions>=64 is<=1.5x its original validation error0.019442108714888683.
- C output recalibration matters: training-only shared-reader writer refit
  reduces its matched-position validation squared error by at least10% relative
  to its frozen function. If B holds and C fails, stable readout is a plausible
  interpretation; if B fails, inspect position/corpus/baseline controls before
  drawing a narrow transfer conclusion. No threshold changes after results.

Price:0nativebodyforwards; nativeMLP FP64 target contractions on58368cachedstates,
5QR+5SVD solves with<=128features; one1153-feature affine normal solve; output
writers saved (~6MB), readers and native background remain charged. Original fit
and capture costs remain prerequisites. Managed lane1,900s alarm. No nonlinear
reader training or claim of new convergence in this screen.
