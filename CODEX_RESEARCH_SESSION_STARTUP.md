# Codex research-session startup and continuation

## Start this program with Astra

The installed Codex CLI is `0.153.4`. On 2026-09-08, `codex update` resolved and successfully
installed that same latest version. At 14:16 UTC, `codex debug models` reconfirmed that this
account's live catalog lists `GPT-6-Astra` under the exact slug `gpt-6-astra` with
`visibility = "list"`. Astra is already the default in `/root/.codex/config.toml`.

In a running Codex terminal session, enter `/model` and select **GPT-6-Astra**. If Astra is absent
from that picker, exit the session and explicitly override the model. To preserve the current
conversation history, use:

```bash
codex resume --last -m gpt-6-astra -C /workspace/tensor_language -a never -s danger-full-access
```

To start a clean session instead, use:

```bash
codex -m gpt-6-astra -C /workspace/tensor_language -a never -s danger-full-access
```

In the ChatGPT desktop app, first use **Menu > Check for Updates**, then create a new Codex chat
and open its model picker. Astra is rolling out gradually, so it may still be absent from the
desktop picker even when the CLI catalog already exposes it. The explicit CLI command above is
the verified path for this account. If the command ever reports an access error, that is an
account/workspace rollout issue rather than a repository or GPU-instance problem.

For a clean session, ask Codex to read this guide and continue the durable research goal.
A resumed session keeps its history but should reread the current pointers and clocks.
The guide remains the restart authority for research reviews and managed runners.
To diagnose a stale picker, query the actual catalog:

```bash
codex debug models | jq '.models[] | select(.slug == "gpt-6-astra")'
```

## Purpose and authority

Read this guide before choosing research work. The durable goal is a simpler
executable explanation with OOD prediction, independent extraction, selective
removal/interchange, and composition/reuse. Structural simplicity means fewer
independently specified computations, including adapters and all opaque weights.
A passed local identity, good selected-token fit, smaller tensor or finished
pilot is not completion.

The user corrected the stale goal wording: follow the
[bilinear reconstruction handoff and appended success criterion](basis_aligned/polynomial_causal/explanations/bilinear_circuit_reconstruction_codex_handoff.md)
and [original pilot report](basis_aligned/polynomial_causal/explanations/bilinear_reconstruction_pilot_report.md),
not better_math_ideas.md. The two unembedding paths are both active: individual
token readers, and shared clusters/hierarchy contrasts/components with their
token-specific remainders, folded backward through the actual model.

## Restore current state

1. Read `/root/.agents/skills/bilin18-research-driver/SKILL.md` completely when
   first applying it. Its continuation, circuit focus, hourly and mathematical
   review instructions remain in force.
2. Inspect the durable goal at the start and before yielding. Classify the
   previous turn as progress, a verified live wait, or no progress. Never mark
   the full goal complete because one rung or report finishes. Before yielding,
   append a board claim and actually execute the next CPU analysis, begin the
   committed next implementation, or leave its audited GPU job queued/live.
3. Read the protocol and latest relevant tail of [AGENT_BOARD.md](AGENT_BOARD.md).
   Claim work before building. Claude shares this checkout and runner.
4. Read the single current result summary:
   [LATEST.md](basis_aligned/polynomial_causal/explanations/2026-09-11/LATEST.md).
   Follow its explanation and primary receipts. Use the
   [explanation index](basis_aligned/polynomial_causal/explanations/README.md)
   when the dated location changes. Do not reread the archived startup history
   unless a specific earlier claim requires it.
5. Inspect recent commits and dirty state in both `/workspace/tensor_language`
   and `/workspace/theseus-bench`; relevant tails of
   [BILIN18_CONNECTION.md](basis_aligned/bilinear_quotient/BILIN18_CONNECTION.md)
   and [BENCHMARK_BACKLOG.md](basis_aligned/bilinear_quotient/BENCHMARK_BACKLOG.md).
   Current files, processes and receipts override older narrative snapshots.
6. Before pursuing an interesting component, check the
   [module index](basis_aligned/bilinear_quotient/modules/INDEX.md),
   [MLP index](basis_aligned/polynomial_causal/explanations/MLP_MODULE_DOSSIER_INDEX.md),
   [module records](basis_aligned/bilinear_quotient/circuits/MODULE_DOSSIERS.md),
   its relevant dossier, aliases and primary receipts. Missing consolidated
   coverage is documentation debt, not evidence the module is unexplored.

## Current handoff — updated 11 September 2026, 04:06

Read explanations/2026-09-11/LATEST.md and explanation_2026-09-11_0406.md.
MLP16_PRODUCER_OVERLAP_V1 managed2.00s: AheldB/Cmissed. FrozenQK/OVsourcefeature
functionoverlap5.81%vsraw4.75%,rotatedcontrols5.33%; positiveexcessmisses.05bar,
noheadtopcos>=.95. Exactlocalresidual/bias/RMS/x0replay3.2e-15. No textfit.
Cache /dev/shm/bilin18_mlp16_producer_overlap_v1.pt, hashinresult.

Redteamexecuted: MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT exactfulljointkeyspace
canonicalalignment raisesmean33.63%,5.79xold. Perheadmaximumcos.672–.843,stillno.95.
Candidate17framescached /dev/shm/bilin18_mlp16_producer_key_envelope_v1.pt.
NextCPUroutingtradeoffaudit executed: 8fixedpositionsquery511,oldQKtouch42.03%,
new16.63%; newframesoptimizeproduceralignment atcosttorouting. No adoption.
NativeG16notlowrank,rank90=848; radialcoefficientfraction.0142%. Checkdossiers.

NextCPUconsequence completed: coupled_producer_routing_objective_v1.py,CONTROL
passesgradient/gauge/finite-difference. Combineexactfunctiontracequotient with
normalizedsourceinfluencetrace on17Dsubspace restrictedtojointkeyspace. Need
registeredtradeoffweight, nativeinfluencepreparation andexistingmanifoldoptimizer;
reuseinsteadnewsolver. Nativejointfitnotrun/queued. ActualQKtouchseparatefrom
influencesurrogate; fullnormalizers/backgroundremain. No datafitting authorized
untilweightstructuralavenuesexhausted. Old B/Cmissespreserved.

All ownGPUjobsfinished at04:06; inspectsharedrunner beforeenqueue. Next hourly
04:22, math04:49. Fullgoalactive. Filesystem filledduringwrite; relocated48MBof
verifiedinactiveSep4temporarycompilefiles to/dev/shm, originalpaths aresymlinks.
Receipt /dev/shm/codex_relocated_inactive_compile_cache_20260911_0403/receipt.json.
About46MBfree afterrepair; do notdeletecurrentexperiments orotheragentstate.
Priorreaderfamilybounds0352 andfrozenFineWebvalidation0338 remaincurrenthistory.

USER PRIORITY CORRECTION, 21:47 UTC: weight-first structural discovery. Exhaust
substantially different weight-only assumptions with appropriate stronger solvers
before incorporating data into discovery. FineWeb is the model's training corpus
per user; Pile is shifted-corpus/OOD validation, and Pile-adapted fits are not clean
OOD evidence for those fitted surrogates. Stop expanding data/CE/Fisher fitting.
Use FineWeb for in-distribution validation, Pile for separately labelled transfer.
Do not substitute more data experiments for the still-mostly-unrun25hypothesis list.
No own data jobs remain queued. Existing completed data results are historical.

Full goal active. User explicitly requests a broad unsupervised structural search,
substantial data/optimization, convergence and red-team review of negatives.
Read explanations/2026-09-10/unsupervised_structure_campaign.md first (25 hypotheses).
Standing user instruction: red-team every future negative using that document's
red-team gate before drawing structural conclusions. Record the narrow failed
claim, strongest plausible methodological explanation and executed discriminating
check; otherwise mark audit pending. Preserve original results and thresholds.
The prior joint32 fit was a short pilot, not a census of two mechanisms.

UNSUPERVISED_DATA_V2_RESULT.json:250forwards1000seq512000processed tokens,
64000stored MLP17 pairs, fixed800/100/100row splits. Decode scaled FP16 as
x.float()*x_scale and y.float()*y_scale before casting to FP64. Native bias key
is transformer.h.17.mlp.Down_bias, not Down.bias. DataV1 fit failed before
optimization on that key; V2 execution repair passed64-state native replay.
Data capture itself remains valid. No fresh/document-level/OOD claim.

Two V1 weight chunks finished but neither converged: product_s0 coefficient
error.91262749; shared_reader_s0 .94666619. Saved checkpoints+CHUNK00 JSON exist.
CPU validation errors on natural states are.093409/.082079 respectively, showing
metric-dependent ranking. Stall audit: product cancellation ratio14264.96, saved
LBFGS step0. Fresh/centered/projected refinements fail1e-8 improvement bar.
Exact pair regrouping reduces cancellation8.17x, misses10xbar, costs130vs128products.
Fixed-reader penalty.01 reduces cancellation to1.245 while retaining80%of captured
coefficient energy. energy_regularized_quadratic_v1.py and8dense/gradientcontrols
implemented; joint native penalized fit has now CONVERGED LOCALLY. New convergent_quadratic_fit_v2 tracks
optimization_loss, reports reconstruction separately and preserves gradient bars.
Planted/resume and nonstationary-stall controls pass. Sources/bindings are frozen.

Penalized weight product V1 completed21:07:47: all3predictionsPASS, localconvergence
in57.04s/3931closures, capture.086989,cancellation1.219, validation.091166.
Original540s warm-start cost remains charged; changed objective not originalfitconvergence.
V2 data_shared_reader_s0 completed, validation.019442, unconverged. Productchunk1
completed21:16:52, train.01621265, stillunconverged; avoidmoreidenticalzero-progresschunks.
Blockvalidation.024394; firstproductvalidation.025849; secondproductvalidation.029441
worsened despite tinytraininggain; firstfunctionfrozen. Affine.031560.
MILLION_TOKEN_PANEL_V1 rows prepared:2048distinct Pile-10kdocumentprefixes,1048576tokens,
32sampledinputs/document,1600/224/224splits. Managedcapture completed21:18:19 in27.74s, all3instrumentchecksheld.
MILLION_TOKEN_PANEL_V1_RESULT.json is authoritative; newdatasetfitpending. Input-onlycache~164MB; all819200trainingpositions
contribute mean/secondmoment. Pile isnotverified pretrainingdistribution; covariancenotfourthmoment.
See appended21:14/21:18campaignanswers foroptimizerdetails, costs and primaryliterature.
QR variableprojection objective implemented and toy first-gradient/SVDcontrols held.
PILE_FIXED_READER_TRANSFER_V1 completed21:26:07 in4.94s, all3predictionsPASS.
Shared-reader frozen/refit Pilevalidation.015637/.014018; affine.023747.
Matched>=64 sharedwriterrefit improves10.83%. No nonlinearreadertraining yet.
QR/SVD andnormal-equation functionbridges held; qr_seconds are asynchronoushosttimings,
not GPUbenchmarks. Writergeometryaudit completed: sharedfunction cosine.98575.
PHYSICAL_QUADRATIC_V1 completed21:36:02: A heldexactly, B/C failed. Shared-refit
MAE tokenCE.23836/top1.87653; mean CE damage+.05879; KL9.81% reduction misses10%bar.
Do not turn localreconstruction into a circuit/prediction claim. Native MLP17
replacedbyactualshared64 execution,238720coefficients vsnative15926400; backgroundretained.
PILE_QR_REFINEMENT_V1 completed: Aheld B/Cfailed; bothunconverged. Validation
normal.0130643/QR.0130386; QR not1%better, fewerclosuresinmatched240s.
TERMINAL_PROBABILITY_REDTEAM_V1 completed: A/Bheld Cfailed. FisherKL predictions
within.64–2.38%; normonly accountsfor.67–1.04%of fullKL. Preserve diagnostic,
but user now defers probability/data-guided fitting until weight-first search exhausted.
Native WEIGHT_PRODUCT_ALS_V1 completed: numericalchecksheld, convergence/referencequalityfailed;120.25s,517sweeps,113698CGiterations,obj.91425633. Redteam shows continuedouterprogress despite accurateinnersolves; no structuralnegative. New multioutput16x16x4 blockweightfit running, source/bindingfrozen. Signed-squarewrapperprepared butnotqueued; checkdiskbeforeexecution. See campaign22:08 formath/prices. Next strongerweight-onlysolvers anddistinctrepresentations; no newdatafit. JointGN core andstepacceptance controls nowpass; WEIGHT_PRODUCT_GN_V1 managedqueued behindblockfit. GNcompleted Aheld B/Cfailed,obj.91422509,stationarity.001625. Blockcompleted Aheld B/Cfailed,capture.0862302. CPUblockgeometry andexactcommon-outputsplit audited: commonchannel7.195%nativeenergy, blockcommonerror.279656, centerederror.962932. Preservecommonbeforetanh. SignedsquareV2completed369.75s,Aheld B/Cfailed,capture.0964521,stationarity.0003955. CPUstoppingaudit confirms maxgrad9.5e-11 belowinternal1e-10 butexternalbarfails; rawrownorms53–2037. Polishnowcompleted17.26s A/B/Cheld, capture.09645222, canonicalstation3.296e-5/original8.095e-5: locallyconverged. Newcenteredorthogonal128inputbasis/256edgebaselineAheld B/Cfailed,capture.01401, fullcore.06406. Stiefel sparseframeupdatecontrols pass; SPARSE_CORE_RCG_V1 native240s nowmanagedqueued, SHAde4b10eb6f52ac3301a70ba9f6147b3528152340fc0f4394d51048ee261cae25. NativeRCGcompleted74.30s A/Bheld Cfailed,capture.0366952,station8.49e-5. Math2249review andsquarepencil/nativeports auditscomplete. Independentseed937restartcompleted83.42s A/Bheld Cfailed,capture.032859, functioncos.795508. Exacttwofunctionspan capture.038967 atdoubledcenteredcost, noadoption. NativegeneralblockQRgaugerepair controlspass, INITIAL.pt saved; nextmanifold-awareoverlappingblockoptimizer onoriginalfull-Ulambda.01objective. No newblockoptimizationyet. See latestcampaignappendix. NoV1squarerun. See campaign22:26.
FullU output-function andtrace red-team audits complete, coefficientmetric-only bounds.

32original configs remain frozen:4representations x2metrics x4starts. Only first
chunks described above have executed. Each540-second fit chunk saves optimizer
state; ending a chunk is not convergence. V1 cores/bindings must stay frozen.
Use corrected V2 runner for data configurations. States~296MB and unfinished
optimizer checkpoints are local-only; most artifacts are not off-box backed up.
Disk~238MBfree after million-token capture; monitor before further queueing.

The previous fixed two-product causal screen held its instrument but failed
sufficiency/selectivity. Math1949 gives tested limits on exact sparse output-token
supports through U, not behavioral impossibility. QK V2/input pullback is a separate
26-head/two-behavior experiment; it is NOT unembedding→MLP17→last-attention folding.
The longer backward-folded path remains authorized and pending in the campaign.
User authorities: unembedding_folding_in_math.md, unembedding_factors_how.md,
and original bilinear reconstruction handoff/pilot; not better_math_ideas.md.

Reusable saved states (avoid recapture):

- `LEXICAL_FORM_INTERCHANGE_V1_STATES.pt`: final states, four token scores,
  structured unembedding readers and their MLP17 product coefficients.
- `TOKEN_CONTEXT_SOURCE_V1_ARTIFACT.pt`: base u17 and all36 module outputs,
  token context readers and MLP16 product pullbacks.
- `GERUND_READOUT_FACTORIAL_V2_STATES.pt`: native/edited terminal states,
  normalized MLP17 inputs and compiled token/norm response states.

Reusable code: `scalar_write_network_executor_v1.py`,
`quadratic_readout_state_v1.py`, `paired_panel_bootstrap_v1.py` and
`bilinear_quotient/circuit_registry_v2.py`. Files used by frozen bindings must
not be changed retroactively. Build small wrappers for new semantics.

## Review clocks and throughput

Latest hourly review:
[03:22](basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-11_0322.md).
Next hourly review is due **04:22 UTC on11September** at the first safe boundary.
Latest mathematical review:
[01:49](basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_0149.md).
Next mathematical review is due **04:49 UTC on11September**. Derive later deadlines from the
newest authoritative review files, not this snapshot. Do not duplicate reviews.

Hourly reviews restate seven circuit targets: explicit computation;
cross-boundary grouping and within-module splitting; held-out/OOD prediction;
extraction/sufficiency; selective manipulation; composition/reuse; stable
identification. Audit changes, confounds, alternative directions, serial
throughput and `CIRCUIT_FOCUS`, `CEREMONY_BUDGET`, `NOVELTY_LESSON_GATE`.
A failed gate forces repair before unrelated work. Aim for one screen/null per
10 serial minutes, with deeper confirmation only after a basic screen passes.

Mathematical reviews define the actual tensors, indices, graph, nonlinearities,
gauges, domain, error norm and literal price. Search primary literature, map
assumptions precisely, and derive an executable circuit consequence. Merely
listing papers or renaming a tensor network is insufficient. Immediately act
on the best consequence.

Use the existing `research_phase_clock_v1.py` and
`RESEARCH_ACTIVITY_2026-09-10_1614.jsonl`. Mark the first tool boundary, including
after compaction; mark implementation and validation separately, publication
BEFORE writing reports/registry, and `turn_boundary` before yielding. Never
invent missing phase times.

Publication repair from18:14: keep the startup guide as pointers/instructions,
not another result ledger. Put new math/results once in the primary explanation;
LATEST gets a short result and link, the board an ownership/verdict/link, and
dossiers only new component-specific facts plus receipt links. Reuse the
existing registry writer and paired scorer; do not build another publisher or
audit framework for each screen.

## Managed execution and shared workspace

Inspect live state before GPU work:

```bash
supervisorctl status bqrunner bqrunner2
tail -n 20 basis_aligned/bilinear_quotient/runlogs/runner.log
cat basis_aligned/bilinear_quotient/queue.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

GPU work goes only through lane1. Freeze predictions, rows, measured bars and
literal price; syntax/dry-run checks and reviewed source hash precede enqueue:

```bash
cd /workspace/tensor_language/basis_aligned/bilinear_quotient
EXPECTED_SHA256=<reviewed-sha256> bash ops/enqueue.sh /absolute/path/to/runner.py
```

Lane2 is CPU-only. Never launch a competing direct GPU process or a duplicate
runner. If Supervisor says the runner is stopped, start that managed service.
An observation timeout is not a terminal job; recheck the same handle/log.
`FORCE=1` is only for a verified execution-only failure with a recorded repair.
For direct CPU analysis use `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2 /venv/main/bin/python`.

Preserve concurrent dirty files. Stage only owned paths, fetch/reconcile remote
changes without stashing or rewriting others' live work, then commit and push
each durable unit. `/workspace` is not volume-backed. The user authorized
in-scope research, writes, managed runs, commits and pushes; do not ask routine
permission. No Slack/email messages without explicit authorization.

## Native model and interpretation reminders

bilin18:18 blocks, residual1152,9 heads x128, bilinear product width4608,
50304 output rows and545902902 parameters. Untied unembedding; RMS uses native
float32 epsilon1.1920928955078125e-7; final score is30*tanh(U*RMS(h)/30).
Attention multiplies two normalized, position-rotated QK scores and has no
softmax. First-layer values are shared with learned signed mixing. Residual
re-entry coefficients are learned, not assumed one. Bilinear MLP includes its
Down_bias; no SiLU. Preserve actual rounded rotary semantics.

Checkpoint is the local Hugging Face snapshot for
`Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd`, snapshot
`ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`, SHA256
`680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
[Architecture contract](basis_aligned/polynomial_causal/BILINEAR_RECONSTRUCTION_ARCHITECTURE_CONTRACT.md)
and `jacclust/tt_model.py` plus current execution code govern exact semantics.

CE added above the real model is damage: lower is better. Absolute change is a
preservation metric; signed improvement does not rescue a preservation failure.
Native module boundaries and a chosen reader basis are proposals, not semantic
units. Weight overlap, low rank and exact folding alone do not establish causal
reuse. Selected-token accuracy is not full-distribution prediction. Keep all
native initialization, background, adapters and arbitrary constants charged.

## Preserved history

The former4,335-line startup was preserved byte-for-byte at
[the18:14 archive](basis_aligned/polynomial_causal/session_history/CODEX_RESEARCH_SESSION_STARTUP_2026-09-10_1814.md).
Its earlier ownership, failures, corrections and inactive directions remain
available. Read that archive only for a specific historical question; current
processes, commits, LATEST and primary receipts remain authoritative.
