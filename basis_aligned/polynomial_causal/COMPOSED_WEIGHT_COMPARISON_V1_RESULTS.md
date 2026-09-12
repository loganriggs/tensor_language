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


### Frozen token readout: shared parent, different output branches

A CPU inspection of the exact self branch and first eight SVD partner branches uses every unembedding row, with no token-based fitting. Each branch computes the same parent scalar times a different partner scalar, then writes its own output direction. Both raw centered token coefficients and cosine-normalized coefficients are reported, so high token-vector norm is not the only view.

Several raw output extremes are recognizable: branch1 has “ was”, “ had”, “ are”, “ have”, “ is” on one side; branch3 has “ vanished”, “ ceased”, “ commenced”, “ saved”, “ stopped”; branch8 has “ making”, “ taking”, “ opening”, “ setting”, “ keeping” against “ tweeted”, “ became”, “ saw”, “ went”. Branch7 includes “ challenge”, “ join”, “ visit”, “ change”, “ modify”. The positive/negative orientation is an arbitrary factor sign; actual effects also depend on the input product's sign. These examples were inspected after fitting and are exploratory, not registered semantic discoveries.

The partner output directions are orthogonal in the centered-unembedding metric to2.4e-14 numerical error, as SVD requires. This is an algebraic property, not evidence of causal independence. Their input partner readers are likewise orthogonal in the producer-whitened coordinates while sharing one parent. Top12 absolute token coefficients carry only0.24–0.67% of each inspected branch's output energy: recognizable extremes coexist with broad vocabulary support. Some extremes are fragments or unusual tokens, and the readout remains conditional on the native normalization/background.

This yields a concrete reuse hypothesis: one producer computation gates several distinct output operations through different partners. Testing it requires predicting branch products on new inputs and measuring selective branch interventions and their joint composition. It does not repair the five failed aggregate context swap cells. [Full raw/normalized token lists](MATCHED_PARTNER_TOKEN_READOUT_V1_RESULT.json), [CPU computation](matched_partner_token_readout_v1.py).


The frozen parent now matches the independent continued frame at cosine **0.9995825**, uniquely above the unchanged0.95 bar (next best0.00924). This is stronger reader recurrence than before continuation; the full fitted programs remain unconverged. [CPU receipt](MATCHED_PARTNER_PARENT_RECURRENCE_V1_RESULT.json). A [frozen branch intervention screen](MATCHED_PARTNER_BRANCH_SCREEN_V1_PREREGISTRATION.md) tests task localization, input-coefficient changes and nonlinear joint effects; it is queued, with no result implied here.


## Frozen branch interventions —12 September06:54

The registered past/progressive conjunction fails, with a useful split. Branch3 (past-token readout) passes localization in0/8contexts; its progressive swap effects exceed its past effects. Branch8 (ing-token readout) passes in8/8contexts: progressive swap RMS0.022–0.032 versus other-task maxima0.00115–0.00190, a14.3–24.5fold ratio. Both have live input-coefficient changes. Joint3+8 writes replay exactly and summed individual swap effects predict joint effects within1.53% in every cell. These are conditional write interventions on an inspected panel, not upstream extraction or OOD evidence.

Signs prevent a simple support label. Branch8's mean progressive donor-swap effect is **−0.02752**, although its mean removal CE damage on that family is **+0.00814**. Removal and interchange answer different questions; neither sign should be silently reversed. A frozen factor-attribution follow-up separates parent, partner and input-normalization changes before interpreting this behavior. [Branch screen](MATCHED_PARTNER_BRANCH_SCREEN_V1_RESULT.json), [attribution mathematics](MATCHED_PARTNER_FACTOR_ATTRIBUTION_V1_PREREGISTRATION.md).

A CPU red-team reused the existing full spelling-pair inventory. Branch3 has95.4%consistent add-ed differences across1008pairs; branch8 has98.9%consistent add-ing differences across921pairs. However, paired difference energy relative to independently paired marginals is0.877 and1.009 respectively. Mean contrast is exactly invariant to pairing. The output pattern is real and broad, but does not demonstrate a lexeme-preserving transformation or explain why the branch's input changes. This is especially clear for branch3: strong spelling readout coexists with failed task localization. [Spelling-pair receipt](MATCHED_PARTNER_SPELLING_REDTEAM_V1_RESULT.json).

An initial execution attempt failed when disk space reached zero, before a Codex result was written. The managed retry completed in42.18seconds after space recovery. That infrastructure failure is separate from the scored scientific miss above.


### Partner change explains the sign better than normalization

The exact coefficient split $\Delta(str)=\bar r\bar s\Delta t+\bar r\bar t\Delta s+\overline{st}\Delta r$ separates partner, parent and inverse-RMS changes. Native margin replay is exact at the reported precision. Partner-only progressive swap effects are negative and5.0–11.9times the parent-only RMS across all8contexts; parent mean effects are positive. Norm/full effect RMS is6.1–20.09%, so the all-context20%bar fails narrowly in one context. Keep that miss: normalization is not negligible, but it is not the sole source of the negative sign. The sum of separate native effects passes the10%error bar everywhere. [Attribution receipt](MATCHED_PARTNER_FACTOR_ATTRIBUTION_V1_RESULT.json).

Next, a [frozen new-construction panel](MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PREREGISTRATION.md) uses16 previously unused verbs across immediate progressive, intervening adverb, gerund complement and quoted distractor. Rows are built, but no native outcomes have been measured. This tests generalization and output-cue confounds without fitting to data.
