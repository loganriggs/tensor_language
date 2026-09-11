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

## Current handoff — 11 September 16:22 matched joint fits

Use [the latest requested report](basis_aligned/polynomial_causal/explanations/for_logan/LATEST.md),
[current state](basis_aligned/polynomial_causal/explanations/2026-09-11/LATEST.md),
and the [method index](basis_aligned/polynomial_causal/WEIGHT_ONLY_METHODS_INDEX.md).

- The output-eliminated LL1 run completed16:01:20:11.8891/11.8510%capture,
  both unconverged; functioncos.90849 and7/64groupmatches missstability.
  Read PROJECTED_LL1_CONVERGENCE_V3_RESULT.json; do not restart this completed run.
- Joint multi-parent graphs now execute correctly. Compatible parent selection
  and converged matched all-core solves leave most approximation loss.
- New all-core variable-projection kernel moves shared/private readers and output
  directions jointly. Dense gradient/FD checks pass. Planted near4/4recover,
  independent0/4; coordinate re-encoding recovers1/4 after cycles.
- Raw parameter norms reachedmillions and caused false gradient stopping.
  Same-function re-encoding exposes a large gradient; raw optimizerrestart doesnot.
  Read SHARED_READER_VARIABLE_PROJECTION_V2_MATH.md before fitting or claiming
  convergence. A new controller must bound/reset redundant coordinate scales.
- Native kernel preflight completed with all bars held at~.28s/evaluation.
  The bounded/re-encoded controller preserves function but still recovers only
  1/4independent planted starts; no global guarantee or absent-structure claim.
- First spectral pair completed: original11.88387%, sharedgraph11.86782%capture;
  gap.000160538 with1.16446%floats saved, but bothunconverged. Postfit12parents
  retain twoeffectiveconsumers and exact joint-removal accounting. See
  SHARED_READER_POSTFIT_INTERFACE_V1_SPECTRAL.json and the primary method note.
- All four matched arms completed by17:42:21. Both price/capture comparisons
  hold, all fits remain unconverged, and zero frozen cross-start node matches
  meet stability bars. See SHARED_READER_JOINT_FIT_V1_AGGREGATE.json and
  SHARED_READER_CROSS_START_V1_RESULT.json. The retained-history comparison completed unconverged and worse than
  its baseline. The original parent1 support-direction screen failed; the
  separate128-prefix suppression/specificity screen passed, while prospective
  input-gating specificity missed. All these screens are completed. Read SHARED_NODE_CANONICAL_BRANCHES_V1_MATH.md. Check
  livequeue/results beforeaction. Allboundhelpers remainfrozen.
- Frozen cross-start correspondence has been scored; descriptive rematching
  does not repair its miss. Do not call current nodes identified circuits.
- Hourly1722 records all three workflow gates held. Read
  SHARED_FACTOR_OUTPUT_MIXTURES_V1_MATH.md for exact output-basis search, full-native
  reader lifting, and graph-node versus global input-removal scope. The queued
  behavioral screen is validation of frozen weights, with no data fitting.
- GLOBAL_READER_REBASE_V2_MATH.md adds exact global single/two-parent interfaces:
  nonorthogonal node deletion uses dual readers and oblique input removal.
  Execution/composition holds, but generic controls prevent a special-circuit
  interpretation of improved cross-start agreement. Costs increase; no adoption.
- Requested fuller reports remain in explanations/for_logan/; latest remains
  the13:27report with CP/LL1 and hierarchy/DAG appendices. General reliable DAG
  discovery and the four behavioral properties remain unfinished. Follow
  RESIDUAL_PARENT_EDGE_V1_MATH.md and livequeue for the current two matched
  closed-component graph refits; older completed runs must not be restarted.

User correction: discover from weights first. No new data/CE/Fisher-guided fits
until distinct weight-only assumptions and adequate optimization have been
examined. FineWeb is the training corpus; Pile is separately labelled OOD.
Historical Pile-adapted fits are not clean OOD evidence for those surrogates.
The [25-hypothesis campaign](basis_aligned/polynomial_causal/explanations/2026-09-10/unsupervised_structure_campaign.md)
remains an idea source; its initial status table is historical. Follow the
bilinear handoff/pilot and unembedding_folding_in_math/unembedding_factors_how
notes, not better_math_ideas. Check dossiers before opening component work.

Every negative or weak positive needs its narrow claim, strongest plausible
methodological explanation and an executed discriminating check; otherwise
mark the audit pending. Do not replace a failed prediction with a later repair.
Local convergence is not global recovery; weight structure is not a circuit.
Earlier snapshots and cache details remain in Git history and primary receipts.

Publication repair at07:22: update detailed narrative hourly, on a major decision
change, or when the user asks. Between those boundaries, preserve primary
receipts and short coordination notes. Do not append a new explanatory essay
or historical handoff block for every small control. No new publisher framework.

## Review clocks and throughput

Latest hourly review:
[19:27](basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-11_1927.md).
Next hourly review is due **20:27 UTC on11September** at the first safe boundary.
Latest mathematical review:
[19:51](basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1951.md).
Next mathematical review is due **22:51 UTC on11September**. Derive later deadlines from the
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
