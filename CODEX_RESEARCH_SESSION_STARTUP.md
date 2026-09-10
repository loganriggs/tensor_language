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
   [LATEST.md](basis_aligned/polynomial_causal/explanations/2026-09-10/LATEST.md).
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

## Current handoff — updated 10 September 2026, 19:45

Full goal remains active. LATEST links the current explanation and primary receipts.
UNSUPERVISED_JOINT32_V1_RESULT.json (19:31): all-token weight-only joint fit,
32 products, two starts. Instrument held; gain and overall stability failed.
Two stable products have pronoun-related loadings. STABLE_JOINT32_PRIOR_V1_AUDIT.json
compares existing §§1583/1589–1591: one product's readers span the old gender axis
at .961 projection length, but the two-product zero-reflection tensor leaves .9035
full-vocabulary coefficient error. Do not call this a newly discovered gender circuit.
Next bounded screen compares the two products' reflection effect on natural contexts.
No sparsity penalty; exact output loadings all dense. Native remainder retained.

CORRELATIVE_JOINT_QK_SUBSPACES_V2_RESULT.json (19:34) compares pre-position joint
QK1×QK2 spaces. A2 own-routing error improves .628 to .149; A1 .157, C .272.
Overall own-routing/selectivity hypotheses still fail. These are the same opened
rows, not independent confirmation; edits act on joint-feature ports, not native
raw-input vectors. Both unembedding and QK directions remain active.

User mathematical authorities: explanations/2026-09-10/unembedding_folding_in_math.md
and unembedding_factors_how.md. Follow unsupervised all-token factor discovery,
then inspect consumers and shared readers; the earlier four-word family was only
an illustrative program test. Sparse output support and computational DAGs differ.

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
[19:14](basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_2026-09-10_1914.md).
Next hourly review is due **20:14 UTC** at the first safe boundary.
Latest mathematical review:
[16:49](basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md).
Next mathematical review is due **19:49 UTC**. Derive later deadlines from the
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
