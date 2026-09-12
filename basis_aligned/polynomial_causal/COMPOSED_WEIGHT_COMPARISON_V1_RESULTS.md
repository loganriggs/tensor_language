# Composed weight methods: optimization, fidelity and metric comparison

The extra optimization produced one passing **developmental component screen**:
the recurring shared reader and its671 selected interactions now reproduce their
exact local reference under every registered write, removal and swap bar. The
complete sparse graph still fails. This is progress toward extraction of a shared
computation, not yet a semantic circuit or evidence for all four user properties.

Three comparisons also prevent misleading conclusions. Faster optimization makes
many more updates but has not converged. A different representation captures much
more coefficient energy at the same cost while behaving less faithfully. Finally,
using the fully symmetric quartic metric does not materially change the frozen
graph's measured coefficient capture.

## Complete-path comparison

All rows below approximate the same bias-free MLP16-producer/MLP17 interaction.
“Capture” is the fraction of the centered paired coefficient norm explained;
“write error” is relative residual-vector error on the128 developmental endpoints.
These are different measures and should not be substituted for each other.

| Method | Coefficient capture | Native write error | Conditional stored floats | Native fidelity |
|---|---:|---:|---:|---|
| First optimized4096edge graph | 9.09% | 9.12% | 20,643,840 | Fails |
| Continued4096edge graph | 10.36% | 8.93% | 20,643,840 | Fails |
| Selected1365trained products, exact writer refit | 39.67% | 24.22% | 20,642,688 | Fails |

The dictionary's random1365product control captures31.24%; selected support is
better under its coefficient objective. Conditional linear solves have residuals
below1e-15 and modest condition numbers7.13–8.96. This is an exact writer fit for
a greedy support, not a globally optimal dictionary. Its much larger coefficient
capture does not preserve these native interventions. It therefore exposes a
representation/capacity confound without winning the circuit comparison.

Native normalization, U, biases and other paths remain background and are
additional to the conditional prices. None is a deployed whole-model reduction.
[Dictionary receipt](COMPOSED_DICTIONARY_OLS_V1_RESULT.json).

## What the optimizer changed

The continuation performed1180/1187updates in about603/605seconds per arm, versus
114updates in each previous arm. Exact edge searches dropped to60/61. The spectral
arm's update rate increased10.37fold; it continued from an existing fit, so this is
not a matched from-scratch algorithm benchmark. Capture ends at10.358%/9.506%,
with full-function cosine0.8203, up from0.6512. Gradients0.000962/0.001652 still
miss1e-6. Both runs hit time limits: gain passes, convergence and stability fail.
[Continuation receipt](AMORTIZED_SPARSE_FRAME_NATIVE_V1_RESULT.json).

The actual native full-graph swap errors are11.67%,37.00%,41.90%,16.84% across
A1,A2,past,progressive; all exceed10%. Removal disagreements also all exceed0.02.
Thus the modest improvement in aggregate write error has not repaired causal
fidelity. [Native receipt](AMORTIZED_SPARSE_FRAME_FIDELITY_V1_RESULT.json).

## The recurring component passes its local screen

The weight-selected node is still node0. Its reader cosine to the previous node
is0.99885; it has671 incident edges and671 active readers. The comparison is with
the **exact full star on this reader**, not with the entire two-layer path.

| Family | Write error | Swap error | Mean absolute removal CE disagreement |
|---|---:|---:|---:|
| A1 | 3.81% | 1.22% | 0.00359 |
| A2 | 2.70% | 7.00% | 0.00329 |
| Past | 3.20% | 2.90% | 0.00340 |
| Progressive | 3.05% | 3.74% | 0.00382 |

Every cell meets the unchanged5%write,10%swap and0.02removal bars. All swap signs
agree and all16pairs per family are live. Overall write error is3.109%. The
component costs14,481,792conditional floats including the native MLP16 input
maps. Its exact reference removal has negative mean CE damage on all four
families; do not label it a useful morphology mechanism from these results.

The fit was determined from weights. This screen used already inspected
developmental rows. Broader context confirmation has now finished with the same frozen program
and bars. All32cells pass write and removal bounds, but12fail swap fidelity
(ten relative-error misses and two additional sign-only misses). The cache was
previously inspected for other programs, so this is not an untouched holdout
or corpus OOD test. Selective
semantic controls, fresh/OOD evidence and joint reuse remain open.
[Context preregistration](SPARSE_NODE_CONTEXT_V1_PREREGISTRATION.md).

## Fully symmetric quartic metric audit

The existing exact four-linear contraction evaluates both the native path and the
frozen first graph. Independent Gaussian and Rademacher quadruples estimate its
actual symmetric coefficient norm without text or factor refitting. Centered
capture is9.144%±0.058percentage points and9.062%±0.058percentage points
(estimated standard errors), versus exact paired capture9.086%. Full-U capture
is11.94%/11.68%, versus paired11.90%. The distribution checks agree; the predicted
25%relative improvement from symmetrization fails.

This limits the metric explanation for this frozen graph. It is not a theorem
that another factorization would have the same ordering under both metrics.
[Metric receipt](SPARSE_GRAPH_SYMMETRIC_METRIC_V1_RESULT.json).

[Context result](SPARSE_NODE_CONTEXT_V1_RESULT.json): six of the12failedcells have native answer-versus-foil capability>=75% on both sides, so weak native capability does not explain away the verdict. Their absolute margin-error RMS values range0.00126–0.00281, with reference RMS0.00855–0.05963. These characterize the failure; they do not change the bars or justify exclusions. [Magnitude audit](SPARSE_NODE_CONTEXT_FAILURE_MAGNITUDE_V1.json).


## Matched optimal partner subspace — 12 September06:44

Keeping the recurring reader and its self-interaction fixed, replace670 separate coordinate partners with670 freely chosen mixtures. A weighted SVD finds the global conditional minimum of squared coefficient error in this class. The executable program still costs14,481,792 conditional floats, including the MLP16 input maps. No text was used in this fit.

Squared coefficient error falls **58.27%**. Developmental residual-write error falls3.109%→2.260%; context error falls3.020%→2.182%. All developmental bars pass. All32 context write/removal cells pass; context swap failures fall12→5, so overall swap fidelity still fails. Remaining cells: past contexts0,3,4,5 and progressive context1. Past context0 fails sign agreement (87.5%); the others exceed10% relative swap RMS. Preserve the original bars and misses.

The measured SVD residual agrees with the sum of discarded squared singular values to5.1e-13 relative error. Exact self preservation, old-coordinate feasibility, native-star identity and compiled execution checks pass. Thus unconverged optimization is not an explanation for this conditional fit's remaining miss. The scope is narrower than all possible structure: the parent stays fixed, the partners are linear in the whitened producer vector, and the loss measures centered-unembedding coefficient error rather than native behavioral sensitivity. Relaxing the coordinate-partner restriction helped substantially, but did not establish a semantic circuit or OOD transfer.

[Primary result](MATCHED_PARTNER_SUBSPACE_V1_RESULT.json), [frozen program](MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt), [registered mathematics and bars](MATCHED_PARTNER_SUBSPACE_V1_PREREGISTRATION.md).
