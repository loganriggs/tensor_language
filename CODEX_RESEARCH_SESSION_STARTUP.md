# Codex research-session startup and continuation

## Local workstation continuation — 14 September 2026

The local checkout is now restored; see
[local runtime and service status](session_recovery/LOCAL_START_2026-09-14.md).
Use `bash session_recovery/local_namespace.sh /venv/main/bin/python ...` for
historical absolute-path scripts. Both managed runners run under the user's
`bilin18-runners.service`. The user explicitly requested active hourly and
three-hour review scheduling: `bilin18-hourly-review.timer` and
`bilin18-mathematical-review.timer` launch serialized, bounded Codex reviews.
Their prompts read the saved research-driver skill in `session_recovery/`;
they do not generate template reviews. Do not create duplicate runners/timers.
The first local native-weight CPU objective screen is
`NATIVE_RETAINED_ERROR_V1_RESULT.json`; consult the current board for newer work.

## Hourly alternating research tracks — 14 September 2026 local directive

At every hourly boundary, switch the primary track between `CIRCUIT` and
`WEIGHT_FOLDING`. The newest hourly review records `ACTIVE_TRACK`; the following
review must choose the opposite. If older reviews lack the field, the first new
track is `WEIGHT_FOLDING`, because circuit work immediately preceded this rule.

Circuit hours continue causal circuit screens, identification, manipulation,
composition, and dossiers. Weight-folding hours trace native computation paths
backward from unembedding, forward from embedding, or between intermediate layers.
They may fold a whole path or only the pieces relevant to one prediction. In
particular, expand bilinear QK and MLP inputs by earlier residual sources so an
attention-output/MLP self term or either ordered cross term can be isolated and
folded through its actual downstream reader. Measure simplicity as executable
program structure and independently specified terms, states, edges, and weights;
quantization is excluded.

Use the [circuit registry](basis_aligned/bilinear_quotient/CIRCUIT_REGISTRY.md),
[computation-path registry](basis_aligned/bilinear_quotient/COMPUTATION_PATH_REGISTRY.md),
[module dossiers](basis_aligned/bilinear_quotient/circuits/MODULE_DOSSIERS.md), and
[MLP dossier index](basis_aligned/polynomial_causal/explanations/MLP_MODULE_DOSSIER_INDEX.md).
Cross-link a path to the circuits and modules that select its terms. Every three
hours, combine the mathematical review with a bounded organization and efficiency
audit: reconcile these indexes, find duplicate code/work, inspect queue/runtime
health, and make one focused safe refactor or organizational repair when justified.

## Replacement-instance entry point — 13 September 2026

The previous instance was expiring. Read the [final compression report](basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-13_final_compression.md), [recovery instructions](session_recovery/README.md), and [copyable next-session prompt](NEXT_CODEX_PROMPT.md).

Run `/venv/main/bin/python /workspace/tensor_language/research_session_startup.py` for a read-only check of packages, checkpoint presence, portable-artifact hashes, queues, Supervisor services, disk and both review clocks. On a replacement instance, restore dependencies/model first, then use `--install-skill --install-runners --start-runners` as needed. `--verify-checkpoint` checks the full model hash. The script preserves existing differing configurations and never restarts a running service. It does not perform unattended research reviews: the active Codex session must perform them.

Keep the checkout at `/workspace/tensor_language` and Python at `/venv/main`; current executors contain absolute paths. The user prefers Astra; use the available model selection interface without treating a particular CLI version as a research dependency.

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

## User direction change — 13 September, interaction compression

The user explicitly requests compression again, focused on composed interactions and retained terms.
This supersedes earlier skill/startup restrictions against compression as the primary research task.
Follow [the three-setting compression plan](basis_aligned/polynomial_causal/explanations/for_logan/interaction_compression_settings_2026-09-13.md): recent MLP9–attention10–MLP10 interaction, recent head17.2–MLP17 interaction, and full-unembedding/MLP17 control. Compare independent versus joint shared graphs; count nodes, edges, stored weights and execution costs. Discover from weights first, validate behavior after freezing. Do not claim compression itself completes OOD prediction, extraction, selective removal, or reuse. Already queued work may finish; unsubmitted background-closure extensions are deferred while this campaign becomes primary.

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

## Current handoff — 13 September 14:34 research cutoff

The [final requested report](basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-13_final_compression.md) supersedes the old September11 fit snapshot. The startup script and prompt were prepared after this cutoff; no new setting2 fit was run during expiry preparation.

- Best setting1 candidate: `SPARSE_COMPLETE_EVEN_FIT_V1_PROGRAM.pt`, key `0.25`. A sparse 1152×64 reader with25% entry pruning and the correct Gram inverse, fitted to the complete two-QK even numerator. Ten starts locally converged. All five corpus aggregates pass on96additional prefixes, with12individual >10% failures and small signed-control exceptions preserved. Composition and the selected producer-pair aggregate bars pass.
- Price:16.31%reader representation /1.806%declared parent interface saved at commonFP32. All64nodes remain. SparseCPU execution1.9–4.4times slower and more memory-hungry than shared dense execution. No fullmodel/runtime adoption.
- Portable validated package: `basis_aligned/polynomial_causal/extracted_circuits/sparse_even_key_producers_8_2_9_8_v1/`. Its native score arrays exactly replay the frozen compressed implementation. Standalone loading passes; actual normalized contexts and native suffix remain external.
- Stronger regular2of4sparsity fails native transfer even after converged learned-maskrefits. Exact coordinate-chart adaptation also losesCPUtiming. Do not rerun these completed jobs.
- Setting2 sparse mixed tensor saves30.81%versusdense folded storage, retains its conditional regional result, fails broader fullhead inputs. Prior shared-write/balanced/private fits do not beat it; allnodesremain. Setting3fullUshared/privatefits improve matched-budget coefficients but remain unconverged and behaviorally fail.
- Next: inspect setting2producer constraints and derive a weights-only complete retained-contraction error objective at matched total cost. Read `INTERACTION_SPARSE_REGIONAL_V1_MATH.md`, `SPARSE_INTERACTION_EXECUTOR_V1_MATH.md`, `INTERACTION_SHARED_WRITES_V1_MATH.md`, `HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md`, `THREE_GROUP_SHARED_DAG_V1_MATH.md`, `SHARED_KEY_VALUE_MOMENT_V1_MATH.md` and relevant dossiers first. Start with an exact CPUcontrol; no new fit is implemented or queued. Do not duplicate existing outputsubspace, coordinatecomplement orlinearVpullback probes.
- Both managedrunners wereRUNNING, bothqueuesempty at14:44UTC. LastresearchGPUjob: `run_sparse_pair_package_native_v1`, completed14:25:37. Idlecanariescontinue. Inspect live state on migration; do not treat this snapshot as a livewait.

User correction: discover from weights first. No new data/CE/Fisher-guided fits
until distinct weight-only assumptions and adequate optimization have been
examined. FineWeb is the training corpus; Pile is separately labelled OOD.
Historical Pile-adapted fits are not clean OOD evidence for those surrogates.
The [25-hypothesis campaign](basis_aligned/polynomial_causal/explanations/2026-09-10/unsupervised_structure_campaign.md)
remains an idea source; its initial status table is historical. Follow the
bilinear handoff/pilot and unembedding_folding_in_math/unembedding_factors_how
notes, not better_math_ideas. Check dossiers before opening component work.

Controlled British/American paired panels must run `regional_cue_row_check_v1.validate`
from polynomial_causal before native scoring. This shared check catches the repeated
“A American” article confound; it is not a general language validator.

Every negative or weak positive needs its narrow claim, strongest plausible
methodological explanation and an executed discriminating check; otherwise
mark the audit pending. Do not replace a failed prediction with a later repair.
Local convergence is not global recovery; weight structure is not a circuit.
Earlier snapshots and cache details remain in Git history and primary receipts.

Publication repair at07:22: update detailed narrative hourly, on a major decision
change, or when the user asks. Between those boundaries, preserve primary
receipts and short coordination notes. Do not append a new explanatory essay
or historical handoff block for every small control. No new publisher framework.

Newest user-directed proposal: [joint composed interaction paths](basis_aligned/polynomial_causal/explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md). It distinguishes earlier output-only folding from the proposed full-U joint sparse/block/DAG comparison, including MLP16 and mixed attention paths. Exact six-path and quartic coefficient oracles are implemented. First joint path fits converged; coefficient stability passed but native replica validation failed. Follow current LATEST and the primary path note before continuing.

## Review clocks and throughput

Latest hourly review: [13 September14:28](basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-13_1428.md); next15:28UTC.
Latest mathematical review: [13 September14:29](basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-13_1429.md); next17:29UTC.
Derive subsequent deadlines from the newest review filenames. If migration makes them overdue, review at the first safe boundary and establish new clocks; do not backfill offline hours. The14:28ceremonybudget failed: mark phases before operations and keep routine work to primaryreceipts plus shortboard notes untilhourly/major/userrequestedpublication.

Hourly reviews alternate `ACTIVE_TRACK: CIRCUIT` and `ACTIVE_TRACK: WEIGHT_FOLDING`, then restate seven circuit targets: explicit computation;
cross-boundary grouping and within-module splitting; held-out/OOD prediction;
extraction/sufficiency; selective manipulation; composition/reuse; stable
identification. Audit changes, confounds, alternative directions, serial
throughput and `TRACK_ALTERNATION`, `TRACK_PROGRESS`, `CEREMONY_BUDGET`, `NOVELTY_LESSON_GATE`.
A failed gate forces repair before unrelated work. Aim for one screen/null per
10 serial minutes, with deeper confirmation only after a basic screen passes.

Three-hour reviews define the actual tensors, indices, graph, nonlinearities,
gauges, domain, error norm and literal price. Search primary literature, map
assumptions precisely, and derive an executable consequence. They also reconcile
circuit/path/module dossiers and audit repeated code, ceremony, handoffs, queues,
and runtime efficiency. Merely listing papers or renaming a tensor network is
insufficient. Immediately act on the best consequence or focused repair.

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
