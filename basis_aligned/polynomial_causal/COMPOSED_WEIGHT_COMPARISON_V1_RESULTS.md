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


## New verbs, new constructions, and joint behavior —12 September07:09

The frozen branch8 negative interchange prediction transfers to **all16 new verbs in each of three constructions**: immediate progressive, intervening adverb and gerund complement. Their swap RMS values are0.01228,0.02753 and0.02498. Native capability passes every endpoint-family bar (93.75–100%). No fitting or capability filtering was used. This supports a broader ing-form-sensitive readout/input operation rather than a requirement that the last input token be an auxiliary. It remains a conditional contribution with a negative swap sign, not an identified progressive-support circuit.

The answer-preserving quotation control **fails**: RMS0.002567 is20.904%of the immediate-progressive RMS, above20%. That original control adds a whole sentence. A separate registered control compares equal-length quoted “they will VERB” with “they are VERBing”, keeping the outside suffix identical. It passes at RMS0.001607,13.09%of the same frozen reference, with100%native capability. This narrows the length/context confound; it does not replace the original failure or prove zero lexical priming. [Fresh result](MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_RESULT.json), [matched control](MATCHED_PARTNER_QUOTE_CONTROL_V1_RESULT.json).

The two branches have opposing signed effects, so the joint operation was evaluated explicitly on the new cached inputs. Sum-of-single swap predictions agree with actual joint3+8 intervention to0.066–0.551%relative error across allfourfamilies; removal-CE additivity errors are below0.000117. All registered joint bars pass. Mean effects nearly cancel in progressive and gerund contexts, but rowwise joint RMS remains nonzero: cancellation of the mean is not disappearance of the computation. This is evidence that these frozen conditional branches compose predictably on this panel. It is not arbitrary whole-model composition or corpusOOD. [Joint receipt](MATCHED_PARTNER_FRESH_JOINT_V1_RESULT.json).

A prior-art output check finds maximum absolute full-U cosine0.344 against the saved earlier canonical branch bank. No close output alias appears in that particular bank; the test does not establish global novelty or compare every earlier input function. [Alias check](MATCHED_PARTNER_PRIOR_OUTPUT_ALIAS_V1_RESULT.json).

The next unresolved identification issue is whether the **whole parent/partner/writer operation**, not only its parent, recurs across independently fitted frames. The shared parent already matches at0.99958, but that alone does not identify an SVD branch. CorpusOOD, standalone input-state production and stronger selective-removal semantics also remain unproved.


## The whole branch recurs across starts —12 September07:14

The fixed same-ordinal branch from independent arm1/node251 passes the coefficient-recurrence and native effect-preservation bars. Paired coefficient-function cosine is **0.998855**, output cosine0.999641 and partner cosine0.999633. This goes beyond the earlier shared-parent correspondence. Fresh swap errors are0.948%gerund,1.005%adverb,0.733%progressive and7.384%quoted, with100%sign agreement; removal-CE mean absolute disagreements are below0.000290. The comparison was selected by weights before these effects were inspected.

The registered5%write criterion still **fails** in three families:5.31%,5.66%,5.31%; quoted control4.71%passes. A CPU error audit reconstructs the difference exactly. Between91.5%and99.6%of its squared error is common to both members of each base/donor pair; only0.45–8.45%lies in their difference. This explains how endpoint writes can differ more than interchange effects, without repairing the failed criterion. Both coefficient and writer changes contribute, with a small positive cross term; it is not a pure output-scale error. [Independent-start result](MATCHED_PARTNER_CROSS_START_V1_RESULT.json), [error split](MATCHED_PARTNER_CROSS_START_ERROR_SPLIT_V1_RESULT.json).

The component now has weight-only discovery, stable whole-function correspondence between two fits, signed lexical/construction transfer, a matched control, and tested conditional composition. Remaining gaps include full input-state extraction, corpusOOD, broader selective removal and equivalence beyond these two unconverged fits. It is a promising conditional component, not a completed model decomposition or promoted circuit.


## Compact executable input fold —12 September07:24

The stable parent and two partner scalars now execute directly as three symmetric quadratics of the normalized MLP16 input, followed by two shared-parent products and output writes. Upper-triangle storage needs **1,994,688FP32coefficients** for both branches, **7,981,317bytes** including artifact metadata, versus10,632,960floats for their standalone native-factor implementation: **5.33times smaller**. One branch needs1,329,408floats, about8times fewer. The fold is algebraically exact; stored FP32coefficient rounding is measured rather than called exact arithmetic.

FP64 evaluation of the rounded artifact gives1.88e-8/2.37e-8 relative write errors and passes native swap/removal fidelity. The independent CPU executor loads only the compact artifact and declared inputs, with no original MLP/model weights. Across fresh and matched-quote families it passes1e-5write-error bars in actual FP32arithmetic (maximum3.60e-6); FP64maximum2.63e-8. [Artifact and native validation](MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_RESULT.json), [standalone executor](packed_quadratic_branch_v1.py), [CPU validation](PACKED_QUADRATIC_BRANCH_V1_CPU_RESULT.json).

The input functions remain dense: parent/partner3/partner8 need483/479/501signed squares for90%Frobenius energy. The top16capture15.5%,19.0%,11.7%. This confirms complexity for these fixed quadratic forms; it does not rule out other arithmetic rewrites or shared structure. No truncation was applied.

The extraction boundary matters. Call `execute(program, x16, denominator)` with normalized MLP16 input and actual squared MLP17 input RMS. It returns separate branch writes; their sum may be used for a joint conditional intervention. The program excludes the other residual/attention/bias paths, upstream input-state production, full unembedding and remaining model. It is an extracted local program at these ports, not a text-to-answer circuit. If the full native background is retained, its MLP16 weights remain necessary; this does not claim an equivalent reduction in whole-model storage.

For K nonself partners of one parent, packed storage is664,128+665,280Kfloats; the standalone native-factor alternative is10,621,440+5,760K. Packing wins throughK=15and loses fromK=16. Thus copying every partner into a separate dense quadratic would discard useful native sharing. The practical representation should mix small extracted quadratic programs with shared producer maps where many consumers use them.


## Natural-text removal and corpus shift —12 September07:33

The frozen compact component was evaluated on144natural endpoints from48documents, with FineWeb and Pile reported separately. Removing branch8 adds **0.00728CE on FineWeb ing targets** and **0.01833CE on Pile ing targets**. Ing-minus-base contrasts are0.00815and0.01948; mean absolute effects on nearby other-token controls are0.00295and0.00293. Registered support/contrast/control bars pass in both corpora, as do joint3+8CE-additivity bars (worst mean absolute discrepancy0.000335). Native-factor/compact writes and physical model-hook replays pass. These are mechanically selected spelling groups, not POS-labelled grammatical operations.

The overall screen retains a failure: Pile ing native top20accuracy is8/24, below the registered12/24bar. FineWeb ing13/24passes; base-form groups pass in both corpora. No low-capability examples were dropped. Consequently this is evidence of signed component-effect transfer under a corpus shift, not completion of the OOD/circuit criteria or evidence that the model reliably predicts every selected target. [Primary natural-text result](MATCHED_PARTNER_NATURAL_TEXT_V1_RESULT.json).

A paired-document audit finds positive ing removal damage in20/24documents in each corpus. Descriptive bootstrap95%intervals for ing damage are[0.00367,0.01103]FineWeb and[0.01107,0.02685]Pile; contrast intervals also remain positive. Leaving out any one document keeps the support and contrast means positive. Effects are larger when the exact next token is not in the native top20, but remain positive on the top20strata:0.00233FineWeb and0.00696Pile. This rejects a single-document explanation and narrows the capability concern; it does not repair the capability miss or remove uncertainty from small conditional samples. [Document audit](MATCHED_PARTNER_NATURAL_DOCUMENT_AUDIT_V1_RESULT.json).


## Input/context controls narrow the interpretation —12 September07:44

Holding the output writer and coefficient multiset fixed,32derangements were tested in each corpus/mode. The registered conjunctions fail. Actual ing-minus-base removal contrasts **do not exceed the global-shuffle95thpercentile** in either corpus. Thus the earlier selective-removal score cannot by itself identify the input computation: a shuffled coefficient population can obtain a similar score with the same output direction.

The conditional control gives different evidence. Permuting whole documents **within each spelling group** preserves group-level coefficient distributions; actual contrasts exceed all32such shuffled contrasts in both corpora. Actual versus null means are0.00815vs0.00509FineWeb and0.01948vs0.01620Pile. There is measurable context coupling within groups, but the stronger registered all-controls criterion still fails. Replacing native coefficients with shuffled ones changes mean CE by−0.000704/−0.000143under global shuffling, and+0.000327/+0.001028within groups (FineWeb/Pile). Only the last meets the registered+.001damage bar. These are balanced spelling-group panels, not unbiased estimates of whole-corpus loss. [Permutation receipt](MATCHED_PARTNER_CONTEXT_PERMUTATION_V1_RESULT.json).

A CPU marginal audit explains one difference between the controls: global derangement raises expected coefficient magnitude on ing targets to1.455times its actual mean in FineWeb and1.217times in Pile. Within-group permutations preserve those means. Global shuffling therefore changes between-group allocation as well as removing within-group context coupling. This is an exact coefficient expectation, not a proof of its nonlinear CE contribution. No group-mean surrogate was fitted or adopted. [Marginal audit](MATCHED_PARTNER_PERMUTATION_MEANS_V1_RESULT.json).

Retain the exact compact program, independent-start recurrence and conditional effect-transfer evidence. Narrow the interpretation: neither an ing-favoring writer nor selective removal establishes a generally helpful standalone contextual mechanism. The next mathematical question is what the parent/partner quadratic reads from the residual and attention sources that produce x16.
