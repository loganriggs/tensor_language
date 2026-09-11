# Nonorthogonal sparse token-function dictionary

Representation: Q_v-Qmean = sum_j A_vj H_j approximately, with512learned
quadratic functions H_j and signed sparse token usages A_vj. Functions are not
required orthogonal; supports can overlap. All50304token functions enter before
fit, represented exactly in their1152-dimensional Hilbert span. No text or
activations. Common function stays explicit. Unlike varimax, the dictionary
changes and its basis is not an orthogonal rotation inside a fixed rank128 fit.

Let G be native product Gram and CC^T=DGD^T. Exact orthonormal function basis
is C^-1 D H, where H contains native products. Token coordinates X=(U-meanU)C.
Normalize X by sqrt(mean row squared norm) for an explicit global objective scale.
Initialdictionary = top512right singular directions of X, computed from X^TX.
Lambda is median across tokens of their ninth-largest absolute initial coefficient.
This deterministic weight-only calibration makes the initial median active count
approximately8. Lambda then stays fixed.512<1152: this is not an overcomplete
dictionary in the whole function span. Its output-rank restriction remains real.

Minimize .5||X-A B||F²/V + lambda||A||1/V, with every dictionary row norm<=1.
Alternate conditional accelerated proximal solves: soft threshold codes; project
dictionary columns (transposed storage) onto unit balls. Each subsolve<=200steps,
stationarity tolerance1e-5; monotone restart, explicit nondecreasing-step status.
Joint stationarity is measured after both updates, not inferred from conditional
convergence. Before accepting convergence, recompute against exactFP64 X and
FP64converted coefficients. Two consecutive FP64checks<=1e-4 required.

One540second resumable chunk; save best and last codes/dictionary and histories
in unique checksummed /dev/shm cache. A chunk boundary is not convergence. No
global optimum, independent-start stability or semantic identification claim.

A instrument: exact native8-token coefficient Gram/embedding and native centered
energy agree<=1e-8relative; finalFP32/FP64 objective difference<=1e-5;
maximum dictionary norm<=1+1e-6. All values finite. CPU exactembedding and known
orthogonal optimum controls pass; nonorthogonal Lasso objective agrees with an
independent constrained smooth solver<=1e-9.
B convergence: two consecutive originalFP64joint stationarity checks<=1e-4.
C quality/sparsity: converged candidate captures>=.30centered coefficient energy
and median active function count<=16. Codes counted exactly nonzero; additionally
report mean and90th percentile. Original initial-function metrics remain visible.

Literalprice:512*1152dictionary coordinates plus sparse codes/indices (dense
working codes50304*512FP32~103MB). Each dictionary function's input quadratic
remains opaque unless further factored. The native product basis or equivalent
explicit atom matrices remain charged; index counts alone are not an executable
compression. Full-U residual/RMS/tanh interfaces remain outside this screen.
No bodyforwards. GPUworkspace a fewGB; best/last cache~212MBephemeral, durable
receiptssmall. Disk is not used for large new checkpoints.

Null/negative audit: unconverged means optimization unresolved. A converged miss
only concerns thislambda,512function budget and local initialization. Check
objective scaling, penalty shrinkage, sparse-code KKT, atom constraints and
initialization before concluding against sparse shared functions. More than one
start and a penalty tradeoff are needed for representation-level conclusions.

References: https://www.di.ens.fr/~fbach/mairal_icml09.pdf (dictionary objective;
we do not claim its online algorithm's convergence theorem for this batch solver),
https://www.ceremade.dauphine.fr/~carlier/FISTA (accelerated proximal method).
