# Known quartic recovery at wide ambient dimension

22 September 2026, 04:33 UTC. Follow-up to normalized-direction control; queued native protocol unchanged.

Five distinct planted structures: generic quartic products; products of squares; within-output shared quadratic factor; fourth powers; cross-output shared quadratic factor. Each has two outputs and two missing quartic atoms per output. Embed the same four-dimensional target and mean into dimensions 4 and 1152 by zero padding. Extra coordinates have independent unit Gaussian variation and no teacher dependence. The exact known factors remain a representational witness in either dimension. This changes the optimization search space, not the target's intrinsic dimension.

Compare Adam lr0.1, default Muon lr0.01, RMS-adjusted Muon lr0.01; two starts; 250 updates; all other defaults retained and disclosed. Share the first four raw initialization coordinates between low- and high-dimensional cases, adding independent Gaussian nuisance coordinates in the wide case. Optimizer methods share the same complete start. Output coefficients are profiled with ridge1e-6, and factors are normalized row-wise in the forward model. Select checkpoints only by fitting objective. Exact Gaussian population value/response and coefficient errors are diagnostic endpoints. The response metric uses correlated probes rho0.5, not native text changes.

Predictions: at least 8/10 wide Adam fits reach 5% value error; at most 2/10 wide default-Muon fits do so; RMS-adjusted Muon reduces median wide error by at least20% versus default Muon. These are testable expectations, not promotion criteria. Report failures and every family/start. The direction-only control does not ensure these predictions hold for a nonconvex quartic objective.

Record normalized-factor rotation and nuisance-coordinate energy. The raw small/wide normalized initial functions differ because added coordinates dilute signal; disclose this confound. Matrix row count stays four, unlike96 in the native residual learner. Success here does not establish native optimization or semantic recovery. No new native optimizer setting is selected solely by this test.
