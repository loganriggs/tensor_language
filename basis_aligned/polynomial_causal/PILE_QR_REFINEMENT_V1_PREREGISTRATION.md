# Matched Pile nonlinear refinement, 10 September 21:35 UTC

Question: does QR variable projection with a direct whitened residual improve
actual nonlinear optimization over the algebraically equivalent normal-equation
scalar objective? Frozen readers and writer-only solves already transfer and
agree numerically; do not promise an improvement from QR alone.

Both arms start from the identical frozen data_shared_reader_s0 CHUNK00 model,
including its parameter gauge. Both train on all51200 sampled Pile training
inputs and bias-free native FP64 bilinear targets. All-U output metric remains
U^T U. No ridge, penalty, truncation, new labels, minibatches or validation tuning.
Use convergent_quadratic_fit_v2 with fresh L-BFGS histories, no Adam,240fitting
seconds each, identical line search/tolerances and convergence checks. Normal
arm first, QR second; fixed order/warmup is a timing limitation to report.

Prediction A: initial objective absolute difference<=1e-8 and full parameter
gradient relativeL2 difference<=1e-6; all toy controls held; each saved best's
direct-whitened training replay<=1e-8. Any exception or ill-conditioning is an
instrument/optimizer failure, not absence of structure. Save per-arm results.
B: QR arm meets unchanged plateau/relative-stationarity<=1e-4/maxgrad<=1e-7 checks
and improves initial training error by at least1% relative. C: QR validation
error is at least1% lower than normal-arm validation at the same240s fit budget.
These are a convergence and practical-benefit screen; neither global optimality
nor an across-restart ranking follows from two dependent starts.

Validation7168sampled positions is evaluated only after each arm; test unopened.
Save resumable checkpoints, best models/writers, validation and timed wall costs.
Body forwards0, nativeMLP contractions on58368cachedinputs, two240s fit budgets,
~40MB checkpoints. Original source/capture costs remain prerequisites. Managed
lane1,900s alarm. If both arms stall or nonconvergence persists, next investigate
input-parameter conditioning, explicit regularization and dampedGaussNewton;
do not extend identical chunks blindly.
