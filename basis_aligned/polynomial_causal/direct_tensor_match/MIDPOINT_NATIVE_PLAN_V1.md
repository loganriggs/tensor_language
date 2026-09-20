# Native joint midpoint output basis — 2026-09-20 23:41 UTC

Previous coefficient spectrum forbids accurate small shared output spaces under independent isotropic intermediate inputs. Test a different, explicitly data-informed metric: exact target outputs on paired native midpoint/source inputs, including the frozen last-MLP RMS denominator. Target includes mm+rm+ma, previous bias excluded. Fit uncentered output second-moment eigenvectors only on original calibration FineWeb32 documents/context64. Evaluate frozen basis on reused SELECTIVE_CONFIRMATION FineWeb32/code16 at context256. Also report centered reconstruction with a calibration mean as a priced constant; use uncentered primary.

pred_a_exact: midpoint vs direct bilinear difference max relative<1e-5, full basis reconstruction residual<1e-10; exact counts80 documents. pred_b_native64: frozen rank64 reconstruction relative error<0.5 on both held panels. pred_c_transfer: both held rank64 errors <=1.5 times calibration error. Null: native dependence does not yield a transferable small output space. Report ranks4/16/32/64/128/256/512/1024/1152 irrespective of bars.

To diagnose dependence, cyclically shift source m by one token within each document while retaining midpoint n and original denominator. This is an artificial mismatch, not a causal intervention or natural counterfactual. Report frozen basis errors on this control; do not train with it.

Zero trainable input model: this is an exact target output-space capacity screen, not a discovered circuit or a fast implementation. Price80native forwards; native teacher retained, projected outputs still require its computation; output frame costs1152*r. Save second moments, means and calibration basis, not full activation dumps. This metric is linear reduced-logit Euclidean error before final RMS and softcap, not behavioral intervention fidelity. Fit and held panel token hashes included.

## Execution scope correction
The shifted control rotates m after dividing by its own token RMS, rather than shifting raw m and applying the recipient denominator. It is therefore a mismatch of normalized source inputs, with donor normalization retained. This deviation affects only the artificial control; native paired outputs and all registered primary predictions are unchanged. Do not interpret this control as a raw-source intervention.
