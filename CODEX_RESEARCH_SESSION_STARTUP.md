# Codex research-session startup and continuation

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

## Current circuit state at 2026-09-07 18:14 UTC

The canonical dossier is
`basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`.

The v15 construction changed the causal graph enough that the old rank-16 five-MLP source program
failed.  Complete native module patches then localized the changed behavior primarily to attention
8, 9, and 11, with several MLP writers also material.  The input-inclusive prefix diagnostic is
quarantined because replacing the whole embedding sequence trivially replaces the prompt.

The exact 27-head atlas is complete and valid:

- result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1_result.json`;
- `L8H1` behavioral signed projection `.40075`;
- `L11H3` `.25460`;
- `L9H1` `.16310`;
- `L9H4` `.15853`;
- all four have rowwise direction fraction `1.0` and are conditionally necessary inside their
  full attention parents.

This is a real within-module split, not a claim that one head is sufficient.  Attention 8 and 11
are nearly additive across heads, while attention 9 has a `.02936` whole-layer-versus-singleton-sum
behavior interaction gap.

The completed v15 cross-boundary adaptive greedy composition is:

- prior:
  `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_cross_boundary_adaptive_greedy_v1.json`;
- runner:
  `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1.py`;
- runner SHA-256:
  `de7906f02a1739c0ef01d08210a68140bfcac3a496a3d62780bfc024423f6954`;
- expected result:
  `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_cross_boundary_adaptive_greedy_v1_result.json`;
- exact price: 174 forwards over all 64 v15 A1/A2/P/C rows.

The job finished successfully at `2026-09-07T18:14:24Z`.  It is valid but returns the registered
`no_small_selective_prefix` terminal.  The full bank reconstructs A1/A2 behavior at `.958/.966`
and final residual at `.981/.988`, but flips 29/32 controls with median KL `2.627`.  Its first
behavior-sufficient greedy prefix already flips 24 controls.  The terminal proves a local greedy
boundary, not a global subset impossibility result.

The greedy order is selected using actual joint A1 behavior subject to P/C selectivity; A2 never
selects the sequence or stopping prefix.  However, A2 contributed to the earlier pooled module and
head candidate screens.  Therefore prediction C is only a no-reselection construction check, not
a pristine held-out identification test.  A genuinely fresh lexical/construction bank is required
before promotion.

## Result-dependent continuation

The active successor should be an exact 32-subset lattice over the low-collateral attention pieces
`L8H1`, `L9H1`, `L9H4`, `L11H3`, and complete `attn:15`.  It tests whether attention-component
composition restores selectivity before splitting the high-gain, high-collateral MLP responses.

That successor is now implemented and queued through the managed runner:

- prior: `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_low_collateral_attention_lattice_v1.json`;
- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_low_collateral_attention_lattice_v1.py`;
- runner SHA-256: `6747d2d5cc85772f1f20dceb3ab9ac6cf665bfdccad590a1ba028e7cac19f192`;
- expected result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_low_collateral_attention_lattice_v1_result.json`;
- exact price: 35 forwards over all 64 A1/A2/P/C rows.

On restart, inspect the result, runner log, queue, and live process before deciding whether it is
queued, running, or terminal.  Never restart from a stale timestamp alone.

When the next result lands:

1. Score every registered prediction and terminal exactly as written; retain failures and nulls.
2. Inspect the chosen sequence, first eligible prefix, A1/A2 behavior, P/C KL and flips, full-final
   residual agreement, and every selected-component exclusion.
3. Update the canonical dossier and `AGENT_BOARD.md`, commit the result and interpretation, and
   push exact owned files.
4. If a selective prefix confirms, freeze it and immediately preregister a genuinely fresh
   lexical/construction confirmation followed by temporal/is-was joint composition.  Translate
   the confirmed response interfaces through the native attention OV and MLP bilinear weight
   tensors to find explicit upstream writers and downstream readers.
5. If no selective prefix exists, do not add arbitrary complete modules.  Use control-conditioned
   pruning and split the causally selected MLP responses into task-defined, gauge-invariant pieces.
6. If behavior succeeds but full-residual agreement fails, preserve that as an observation-map
   mismatch and build the finite multi-environment causal-response operator needed for the next
   constrained-DAS objective.

Before ending any new session turn, require a continuation receipt: an append-only board claim and
either completed CPU analysis, a committed preregistration with implementation underway, or an
audited job present in the managed queue/runner.

## Suggested first prompt in a new session

```text
Read /workspace/tensor_language/CODEX_RESEARCH_SESSION_STARTUP.md and resume the durable
bilin18/Theseus circuit-finding goal from authoritative current state. Keep the hourly circuit
reviews, three-hour mathematical reviews, and Supervisor-managed bqrunner workflow active. Do not
stop at a result boundary; interpret it, record it, and begin the evidence-selected successor.
```
