# Joint producer sharing and routing influence — native weight fit

Rank17 feature span inside each attention17 head's full256-dimensional joint
key-weight row space. MLP16 producer metric and frozen value readers come from
MLP16_PRODUCER_OVERLAP_V1_RESULT; original routing and producer-optimal source
frames are unchanged. No text, activations or behavioral labels enter the fit.

Optimize the mean of two scores, each divided by its attainable per-head maximum:
1. exact producer-function mean squared principal cosine with frozenOV17space;
2. source-influence trace, averaged with equal position weights over query511,
   even source positions0..510 (the existing discovery split).
The mixing weight is fixed0.5. Producer maximum is the exact full-key envelope;
routing maximum is the leading17-eigenvalue sum of the normalized influence.
This is a nonlinear Grassmann objective, not an exact spectral solution.

Use checked objective/gradient with Pymanopt2.2.1 Grassmann conjugate gradient,
Polak-Ribiere updates and backtracking. Four starts per head: originalrouting,
produceroptimal, and independent Gaussian frames fromseeds1409/1423. Per-start
limits60seconds/5000iterations/50000cost evaluations; gradient norm1e-7 defines
local convergence. A limit or tiny step without that gradient bar is unconverged.
Save finalframes and fitdiagnostics even when convergence fails. No automatic
semantic-negative interpretation. Best frame perhead maximizes this fixed
objective, not subsequent validation or a separately selected metric.

Managed GPU stage prepares compressed matrices and checks prior frame/ceiling
replays; then CPU fits256x17frames using the standard solver. NativeGPU receipts
and fitreceipts separate. Install dependency with uv pip install --no-deps
pymanopt==2.2.1. Source/cache hashes frozen before native execution. No giant
vocabulary tensor or new model-state capture. /dev/shm stores matrices/frames.

Predictions for the completed fit:
- A: preparation controls pass, finalframes orthonormal1e-10 and in keyspan;
  source coefficients and prepared sharing scores replay prior endpoints1e-8;
  scores finite and each run finishes no worse than its start within1e-10.
- B: all36starts meet normalized Riemannian gradient norm<=1e-7.
- C: bestframe producer-overlap mean across9heads at least2x the originalframe,
  while actual jointQK numerator TOUCH on held-out odd sourcepositions1..509
  retains at least80%of originalframe mean (ratio of the nine-head means).
- D: for every head, all starts have mean squared raw-subspace principal cosine
  >=.99 to its bestframe. Stable subspace is not semantic identification.

Report actual inside/mixed/touch separately from the influence surrogate, both
discovery and validation means, and full per-head tradeoffs. Normalizers, base
stream, residual interfaces and native background remain required. This is a
candidate cross-layer feature decomposition, not physical adoption or a task
circuit. Price9*1152*17sourceframe numbers; no claimed replacement saving.

If C misses, check whether exact achieved-objective improvement merely shifted
the tradeoff, whether the influence surrogate misranked true touch, and whether
nonconvergence or start instability explains it. Preserve weight0.5 and all bars.
Any revised objective requires a new preregistration, not changed scoring.

[Pymanopt solver documentation](https://pymanopt.org/docs/stable/optimizers.html).
