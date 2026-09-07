# Codex research-session startup and continuation

## Start this program with Astra

The installed Codex CLI is `0.153.4`. On 2026-09-07, `codex debug models` confirmed that the
account catalog lists `GPT-6-Astra` under the exact slug `gpt-6-astra`, and a direct model call
succeeded. Astra is now the default in `/root/.codex/config.toml`.

An already-running session does not inherit a changed default. Use its model switcher if Astra is
visible; otherwise exit it and start an Astra session explicitly:

```bash
codex -m gpt-6-astra -C /workspace/tensor_language
```

Then paste the prompt in **Suggested first prompt in a new session** at the end of this document.
To diagnose a stale picker, query the actual catalog:

```bash
codex debug models | jq '.models[] | select(.slug == "gpt-6-astra")'
```

## Purpose

This document is the restart handoff for the bilin18/Theseus mechanistic-interpretability
program.  A new Codex session should use it to resume the actual research loop rather than
merely summarize the previous session.

The durable objective is to produce a smaller transparent tensor program that is jointly:

- predictive on fresh and out-of-distribution text;
- composable when task programs or replacements are installed together;
- selectively manipulable under removals, swaps, and edits;
- simpler under literal storage, compute, edge, state, and program pricing.

The current circuit-scale priority is to identify high-quality causal circuits and reusable
circuit-finding machinery.  Low rank, activation reconstruction, variance preservation, or
compression alone is not circuit evidence.

## Required startup sequence

1. Work in `/workspace/tensor_language` and read
   `/root/.agents/skills/bilin18-research-driver/SKILL.md` completely.  Follow its
   anti-pause, circuit-focus, queue, review, and continuation rules.
2. Read only the current state slices first:
   `AGENT_BOARD.md`; the first current entry in
   `basis_aligned/polynomial_causal/explanations/README.md`; the tail of
   `BILIN18_CONNECTION.md` and `BENCHMARK_BACKLOG.md`; the newest hourly and three-hour
   reviews; recent Git commits/status; and the managed-runner state.
3. Treat the worktree, result files, queue, process handles, and Git history as authoritative.
   The prose below is a locator, not permission to ignore newer evidence.
4. Inspect the managed services and queue before doing GPU work:

   ```bash
   supervisorctl status bqrunner bqrunner2
   tail -n 30 basis_aligned/bilinear_quotient/runlogs/runner.log
   sed -n '1,30p' basis_aligned/bilinear_quotient/queue.txt
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```

   Both services are configured with `autostart=true` and `autorestart=true`.  Do not launch a
   duplicate runner or a direct GPU process.  If an authoritative Supervisor status says a
   runner is stopped, start that Supervisor service; do not replace it with `python ... &`.
5. Queue GPU experiments only with:

   ```bash
   cd /workspace/tensor_language/basis_aligned/bilinear_quotient
   EXPECTED_SHA256=<reviewed-runner-hash> bash ops/enqueue.sh /absolute/path/to/runner.py
   ```

   Use `FORCE=1` only to retry the same scientific experiment after a verified execution-only
   failure and an explicitly recorded instrument repair.
6. Preserve other agents' dirty files.  Stage exact owned paths only, commit each durable unit,
   and push it: `/workspace` is not backed by a persistent Vast volume.
7. Do not ask for routine permissions.  The user has explicitly authorized in-scope research,
   repository writes, managed execution, commits, and pushes.  Ask only if genuinely new
   authority or a materially different external action is required.

## Periodic research clocks

At the first safe boundary after each clock expires, perform the review and immediately take its
chosen action.  A reminder or review without a concrete continuation is not progress.

### Hourly circuit review

Use the timestamp in the newest
`basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_*.md`; do not duplicate a review inside
the hour.  Restate the seven circuit interpretation targets: computational specification;
cross-boundary grouping and within-module splitting; held-out/OOD prediction; extraction or
sufficiency; selective manipulation; composition/reuse; and stable identification.  Then audit
the full goal, new evidence and corrections, confounds, alternative approaches, ranked next
moves, serial candidate throughput, and these exact lines:

```text
CIRCUIT_FOCUS: PASS|FAIL
CEREMONY_BUDGET: PASS|FAIL
NOVELTY_LESSON_GATE: PASS|FAIL
```

A failed line forces the next bounded block to repair that workflow before unrelated research.

### Three-hour mathematical review

Use the newest
`basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_*.md` as the clock.  Define the
actual model and circuit target as tensors, dimensions, contractions, polynomial degree,
symmetries/gauges, allowed inputs, preserved outputs, norms, and literal prices.  Map candidate
theorems or algorithms object-to-object, list their assumptions and our violations, and derive at
least one executable circuit consequence.  Begin the best consequence immediately after writing
the review.

## Current circuit state at 2026-09-07 21:07 UTC

The canonical dossier is
`basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`.

The v15 construction changed the causal graph enough that the old rank-16 five-MLP source program
failed. Complete native-module patches localized the changed behavior primarily to attention
layers 8, 9, and 11. Exact head patching then identified a distributed four-head core: `L8H1`,
`L9H1`, `L9H4`, and `L11H3`. These are real within-module splits, not singleton-sufficiency claims.

Aligned, capability-qualified P/C controls repaired an earlier absolute-token-position defect. The
repaired exhaustive five-piece attention lattice is valid and shows that complete head responses
recover A1/A2 behavior (`.80535/.86829`) but are not selective: P has five flips and C has three. A
cross-fitted linear complement removes P collateral but also collapses A1/A2, rejecting simple
linear task/nuisance separation.

The exact pattern/value/interaction decomposition is complete. Its mandatory shared lesson is that
an absolute downstream clamp differs from adding a donor-minus-base delta to a live head already
changed by upstream interventions. The valid absolute-clamp result localizes most target transfer
to value content; pattern and interaction pieces are comparatively selective but too weak.

The latest completed screen is:

- prior: `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_head_factor_dual_greedy_v1.json`;
- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_head_factor_dual_greedy_v1.py`;
- runner SHA-256: `8c02ee2a04faad82c3341667f457c30537f351c9147a26dadd7b19736a5d3a48`;
- result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_head_factor_dual_greedy_v1_result.json`;
- result SHA-256: `51c705281cc4fa10aa7d5c3c54bc3b1ee1e9d6d3af3e74f3b98cef3a34ad0285`.

It is mechanically valid at 152 observed forwards and returns
`no_selective_dual_greedy_program`. The strongest zero-flip/low-KL visited arm reaches only
`.13763` A1. Target-first combinations reach `.84510` A1 but still flip five P and two C rows. The
full 12-factor arm replays its parent within `4.62e-7`, reaches `.80535/.86829` A1/A2, and flips
five P plus three C rows. This null is path-local, not an impossibility theorem.

## Active continuation: target-feasible regularized DAS

Use the protocol derived in
`basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2026.md`:

1. Make cross-fitted A1 transfer a hard feasibility constraint rather than optimizing a soft
   target-minus-complement score.
2. Among feasible projectors, minimize worst-group P KL plus Gaussian-noise/Jacobian sensitivity
   and cross-fold projector instability.
3. Select dimension, noise, and regularization on opposite A1/P parity only; keep A2/C sealed.
4. Use orthonormal projectors and a fixed unit intervention dose so a zero-effect complement cannot
   win by construction.
5. If the subspace passes, translate it through native QK/OV and downstream weight tensors to
   identify explicit writers/readers, then run fresh held-out manipulation.
6. If no target-feasible projector exists, preserve the null and change the object to nonlinear or
   input-conditional subspaces rather than another complement-only optimizer.

For every completed result, score the registered predictions exactly, preserve nulls and invalid
instruments, update the dossier and `AGENT_BOARD.md`, commit and push exact owned files, then begin
the evidence-selected successor. Before ending a turn, require a continuation receipt: an
append-only board claim and either completed CPU analysis, a committed preregistration with
implementation underway, or an audited job in the managed queue/runner.

## Suggested first prompt in a new session

```text
Read /workspace/tensor_language/CODEX_RESEARCH_SESSION_STARTUP.md and resume the durable
bilin18/Theseus circuit-finding goal from authoritative current state. Keep the hourly circuit
reviews, three-hour mathematical reviews, and Supervisor-managed bqrunner workflow active. Do not
stop at a result boundary; interpret it, record it, and begin the evidence-selected successor.
```
