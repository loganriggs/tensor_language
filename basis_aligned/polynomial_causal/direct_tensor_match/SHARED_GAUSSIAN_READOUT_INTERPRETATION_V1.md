**Mean/covariance matching helps the cheaper shared hierarchy substantially, but its current dictionary still misses component fidelity.**

Frozen144quadraticfeatures/512learnedrootpairs, two starts, same1088products1353728coefficients1024indices. Exact noncentralGaussian candidate moments and native cross, no empirical output fitting. Runtime39.39seconds, peak1.60GB. Numerical integrity and registered value tradeoff PASS; registered10%component gate FAIL.

|Fit|Text error,1101/1102|Sampled coefficient error|Root1 same-token response error|Root1 sensitivity error|
|---|---:|---:|---:|---:|
|Coefficient optimal|65.67 /64.09%|96.83 /97.20%|51.14 /50.07%|108.11 /107.74%|
|**Primary coefficient budget1**|**21.06 /20.00%**|**98.89 /99.42%**|**36.60 /38.15%**|**40.13 /42.41%**|
|Unconstrained shiftedGaussian|13.00 /14.57%|281.86 /357.30%|30.42 /26.07%|27.57 /29.40%|

Budget1 is relative to captured regularized coefficient score, not total teachererror. Sampled coefficient errors are not fullnormcertificates. The Gaussian reference is N(calibrationmean,calibrationcovariance); “standardGaussian error” in the rawreceipt is a separate isotropic evaluation. There is no claim that actual activations are Gaussian. UnconstrainedGaussian fitting can greatly damage coefficient fidelity even while improving conditional values. PhysicalFP32 replay is<1.6e-7.

The shared representation has lower cost than mixedCP1536products/2385920coefficients, but worse conditional fidelity: the jointly learned mixedCP reaches6.32/6.59%text error and considerably better finite-removal predictions. These are not matched-training comparisons; only the readouts were learned here. This result does not establish that shared hierarchies are intrinsically worse than CP.

**Executed CPU capacity/transfer audit** uses the same fixed feature dictionary with noGaussianassumption. Root1 readouts are fitted by SVDleast squares, with the existing normalization and strict row-hash/pairreference checks. Calibration fitting uses6144states; the evaluation-fitted oracles use2048openedstates and are never exported.

|Readout fit|Evaluation sensitivity error,1101/1102|Same-token response error|
|---|---:|---:|
|Calibration uniform|21.92 /23.89%|18.91 /27.43%|
|Calibration sensitive|20.34 /21.63%|28.62 /41.83%|
|Evaluation sensitive **oracle**|9.70 /10.79%|25.74 /26.51%|
|Evaluation sensitive + exact pair constraints **oracle**|9.93 /11.01%|0.000355 /0.000355%|

For seed1102, even the unconstrained evaluation-weighted least-squares optimum is10.79%. At the statedFP64numerical cutoff this is a finite-panel linear-dictionary lower bound: no readout of those fixed512features can attain10%sensitivity error, even if given the evaluation answers. Pruning to a subset cannot improve that optimum. This is not an interval-certified bound, nor a limit on different products, different feature directions, allsharedgraphs, globalFrobeniuserror or freshdata. For seed1101 a joint finite-panel solution exists justbelow10%, but calibration fitting doesnottransfer. Neitheroracle is a predictive result.

Consequence: stop readout-only sweeps for this dictionary. Change selectedproducts or quadraticdirections. A direct next structural comparison is to use the sameexact swap machinery with the coefficient-plus-Gaussian metric, preserving the same pool/budget andliteralcost. The preceding swaps optimized coefficienterror only, so their failure did not test thiscombination. Changingdirections is a separate, moreexpensive follow-up; bounded-batch derivativeaccumulation can retainsharing andcontrolmemory if needed.

The newmomentinstrument is itself useful: it computes shared-feature selfGram from15partitions andlowrankquadratic cumulants instead of the8192expandedCPterms, and the teacher-cross expands only256terms at a time. Ten zero/nonzero-mean densequadrature controls verifyvalues andgradients. This is a fitting capability, not a discovered semantic circuit. Nativefinite-removal tests have not been run for these inaccurate shared readouts; no selectivecapitalization, OOD, stableidentity, extraction or compositionclaim.

[Native result](SHARED_GAUSSIAN_READOUT_NATIVE_V1.json) · [Preregistered plan](SHARED_GAUSSIAN_READOUT_PLAN_V1.md) · [Capacity audit](SHARED_RESPONSE_CAPACITY_V1.json) · [Moment derivation and implementation](SHARED_GAUSSIAN_MOMENTS_DERIVATION_V1.md).
