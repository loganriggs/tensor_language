# Does matching the actual first two input moments repair the gain prediction?

2026-09-20 22:04 UTC. The exact calibration-Gaussian objective predicts positive
gain for every canonical mode, including mode1, for both skip families. Yet the
quadratic rank2 skip increases mode1 polynomial MSE by21% on the reused64-token
panel and22% on the256-token panel. This is not simply sacrificing that mode in
the Gaussian objective. Different native intervention panels and nonlinear
metrics must not be conflated with this same-panel polynomial comparison.

Freeze the quadratic rank2 program, output basis and calibration centering.
Recompute exact Gaussian gain under a separate mean/covariance matched to each
of those two actual input panels, without fitting or changing any model.
Compare each predicted modewise gain with its actual same-panel gain. This
separates first-two-moment shift from missing higher-order input moments.

For delta=W(p-c), q=Ep-c, K=Cov(F-base,p), G=Cov(p,p), b=E(F-base),
expected improvement is2<tr(WK^T)>+2 b^TWq-tr(WGW^T)-||Wq||².
For each fixed mode replace output vectors/readouts by their scalar projections.
Keep the b,q terms: centering is still calibration-fixed, not silently refitted.
Exact native and student means/cross moments use existing derivative machinery.

Registered diagnostics:
- pred_a_identity: same-moment calibration replay matches archived Gaussian
  mode gains within1e-6 relative; independent small dense quadrature for the
  shifted-centering formula within1e-10.
- pred_b_sign: mode1 gain remains positive under both panel-matched Gaussians,
  despite negative actual gains (hypothesis: higher moments matter).
- pred_c_gap: on both panels, |Gaussian gain-actual gain| exceeds10% of the
  original program's actual mode1 MSE. Report the raw values regardless.

All panels reused diagnostic data. This is a measure-identification test,
not output fitting, new OOD validation, or a claim that covariance is sufficient.
No broad additional readout sweep until this ambiguity is resolved.
