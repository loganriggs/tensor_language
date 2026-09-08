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

## Authoritative restart delta at 2026-09-07 23:00 UTC

### Live continuation at 06:55 UTC on 2026-09-08

The four-head temporal/is-was program has now passed the confirmation stages that were still
pending in the 05:55 handoff:

- `L9H1 + L9H4 + L11H3 + L15H5` passed simultaneous joint composition on the original bank;
- the frozen dual-command OOD authority passed native capability in 31/32 cells at 8/8 and one
  cell at 7/8, then the same four-head program passed the prospective OOD joint-composition bars;
- a registered H4 midpoint clamp selectively removed the intended half-command effect on original
  and OOD rows with unit reduction, direction preservation, low collateral, and additive joint
  removal.  This upgrades the H4 addition from a predictive screen to a manipulable program
  component for this intervention family;
- the reader-factor split showed that `L11H3:v` is the strong stable reader, while the proposed
  `L15H5:q/q2` explanation failed: q is anti-causal, q2 is weakly positive, and their combination
  largely cancels.  Preserve that falsification; do not use the dominant L11 effect to rescue the
  L15 interpretation.

The source-localization receipt,
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json`,
returned terminal `invalid`.  Its region selections (`bridge` for temporal and `postcue` for
is-was) were initially unclaimable because prediction A failed: the runner compared a parent-relative
full-effect cosine to an older command-gold cosine, producing an apparent replay error.  Preserve
the immutable invalid receipt and do not change A or retro-pass it.  The separately registered
three-forward replay audit has now passed with zero like-for-like error and exactly reproduced the
`.028743869178895265` mixed-target error.  It licenses interpretation of the already-frozen B--E
outcomes: the value-source partition is compositional and stable on OOD text, but task typed;
temporal selects the unchanged bridge and is-was selects postcue.  The immediate continuation is
the frozen query/key routing-partner test for those two regions.

At this checkpoint `gpt-6-astra` is the configured Codex default and is visible in the account
catalog.  Both `bqrunner` services are Supervisor-managed and healthy; their queues are currently
drained after the localization job.  The latest clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0615.md` (next review after 07:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0526.md` (next review after 08:26 UTC).
Inspect Git, files, runner logs, and queue state before acting because they remain authoritative
over this prose.

### Live continuation at 05:55 UTC on 2026-09-08

The composition branch has advanced past the 04:42 prerequisite state:

- physical temporal Q8 and both is-was planes form a stable task-typed union rather than one shared
  subspace (`temporal_iswas_common_final_gauge_basis_capture_v1_result.json`, terminal
  `task_typed_direct_sum`);
- a complete exact-weight atlas finds shared physical interfaces despite distinct states: L9H1,
  L11H3, and L15H5 writers plus L11H3:v and L15H5:q/q2 readers;
- the first dual-command bank is a preserved native null, while the prefix-preserved successor
  licenses all 32 same-sequence rows;
- no singleton parent module reaches 0.50 recovery, so that result remains a registered null;
  nevertheless L9H1+L11H3+L15H5 is a licensed distributed program, recovering temporal
  `.706/.752` and is-was `.523/.491` on FIT/HOLDOUT with at most 1.13% cross-command interaction;
- the frozen greedy extension selects only L9H4, raising recovery to temporal `.833/.848` and
  is-was `.696/.669`. L8H1 is not selected because the four-head arm already meets the prospective
  quality bars with fewer additions.

The current decisive job is hash-bound in managed lane 1 behind live unrelated `v252`; do not
enqueue a duplicate:

- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_four_head_union_joint_composition_v1.py`;
- reviewed SHA-256: `aeee380e62576074ba0ad31217506a1dd48b226a2d3720f0dd05580f58c03e18`;
- result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json`.

It must reproduce the selected four-head single-command metrics, then confirm T/I/TI composition
with <=10% interaction, >=.99 additive cosine, <=.05 simultaneous recovery loss, <=.01 collateral,
and exact later-to-earlier causal zero. Passing promotes L9H1+L9H4+L11H3+L15H5 as the higher-quality
joint program. Failure preserves the already-licensed three-head program and closes L9H4 for
simultaneous use. Do not change the union or bars after the result.

Latest reviews are `HOURLY_STRATEGIC_REVIEW_2026-09-08_0515.md` (next after 06:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0526.md` (next after 08:26 UTC). Latest branch commits
at this handoff include `ee991bf65` (H4 joint runner), `743b24c52` (greedy result), and
`6914642c6` (licensed H3 joint composition). Both bqrunners remain Supervisor-managed and healthy.

### Live continuation at 04:42 UTC on 2026-09-08

The latest strategic decision is no longer router feature engineering. Three increasingly explicit
objects—tied-embedding routing, contextual Gram moments, and projective response shape—failed
stable selective routing, including catastrophic A1/A2 exchange on the v16 construction. The
04:15 hourly review closes that branch and redirects to simultaneous task composition and physical
weight-readable command coordinates.

Two prerequisite jobs are hash-bound in managed lane 1 behind the unrelated live `v248` run, in
this order; do not enqueue duplicates:

1. `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_common_final_gauge_basis_capture_v1.py`
   at SHA-256 `a2bdc76d61c54e6879d84a0a5b451ae39ed798bcea74fdb77e10579ee08d78cd`.
   It stores the temporal 1152x8 Q8 basis and both cross-fitted is-was 1152x2 bases in the same
   physical final-residual gauge and decides shared state versus a task-typed direct sum.
2. `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_dual_command_native_capability_v1.py`
   at SHA-256 `0ebc5d1bd8cbcde0db8bee0aa09e7a5384ce35f58df0002f87e4fb0809d689f9`.
   It uses one model forward for 128 same-sequence 2x2 endpoints and scores temporal plus is-was
   auxiliaries separately in 32 hard capability cells. No rows or templates may be filtered.

The exact joint-command Möbius/additivity scorer is already implemented in
`basis_aligned/bilinear_quotient/ops/joint_command_composition_contract.py`. A causal joint
factorial is eligible only if the native capability job issues its all-row license and the basis
capture is mechanically valid. Read and publish both immutable outcomes first, then choose the
shared-basis or task-typed intervention exactly as the basis terminal directs. If native capability
fails, preserve the null and do not tune or filter this bank post hoc.

Current commits are `264af0b60` (dual-command native gate), `cb75e4b2d` (dual-command authority),
and `2ccef52ef` (joint composition contract). The latest clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0415.md` (next due after 05:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0226.md` (next due after 05:26 UTC).
Both bqrunners should remain Supervisor-managed and automatic; verify them at startup as specified
above, and immediately restore queue depth if either lane is unexpectedly empty.

### Live continuation at 00:38 UTC on 2026-09-08

The current decisive job is already hash-bound in managed lane 1; do **not** enqueue a duplicate:

- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py`;
- reviewed SHA-256: `4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3`;
- current queue relation at this checkpoint: depth one behind live `v244`;
- result path when it lands: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json`.

This no-refit factorial asks whether each stable construction-specific oracle retains its own
target transfer when layer-15 attention is live.  Predictions B/C are the admission gate.  If
either fails, close the L15H5 causal-reader hypothesis for this intervention family: the static
weight alignment was not on the exercised native route.  If both pass, execute the already
preregistered full layer-15 head/module mediation atlas, not an immediate Q/K/V claim:

- prior: `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1.json`;
- pure intervention/accounting contract: `basis_aligned/bilinear_quotient/ops/head_response_mediation_contract.py`;
- contract commit: `223369b3f` (seven focused tests pass).

The atlas crosses upstream off/on with absolute downstream-response source off/on for the whole
layer-15 attention module and each of its nine heads.  It separately measures rescue, reset loss,
bypass, and interaction.  Only a passing singleton-composition test licenses a parity-cross-fit
greedy head union; otherwise use an interaction-aware head-set test.  Only a passing L15H5
head-level result licenses its later Q/K/V split.

The latest review clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0015.md` (next safe-boundary review after 01:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2326.md` (next after 02:26 UTC).  Both `bqrunner` and
`bqrunner2` were healthy at this checkpoint.  Re-check files, process state, queues, and newer Git
commits on restart because those facts can advance after this document is written.

### Live continuation at 23:30 UTC

V16 capability has landed.  The immutable first receipt reports `null` only because it asked for
24 jointly capable rows from 16-row A panels; all eight direction-by-side capability cells are
actually `1.0` and both A1/A2 have 16/16 jointly capable rows.  A hash-bound zero-model audit
translated the registered 24/32 ratio to 12/16 and returned `manifest` without causal access.

The fixed-rank multi-construction causal executor is now hash-bound in managed lane 1 at SHA-256
`8c14405674b4773303e197be04668f29ca97d2662dbeccc0f9653c119716c556`, behind live v238 and the
previously queued v247/v249 jobs.  Its prior, joint fit/selection core, sealing tests, and no-model
preflight are committed.  It fits/selects entirely on v15 A1/A2/P/C, then opens v16 A1/A2/P once;
v16 C remains excluded.  The 23:26 mathematical review also implements an exact projective-bisector
falsifier for the failure branch.  The latest strategic and mathematical clocks are now
`HOURLY_STRATEGIC_REVIEW_2026-09-07_2315.md` and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2326.md`.

The multi-construction run has since completed validly at result SHA-256
`0eba172c8ada3e3cfcaa2826afa7a348480a6b61bde819c1a0daad9ed8dcb1a4` with terminal
`fixed_projector_infeasible_on_observed_constructions`.  No initialization passes both target
constructions in both held parities.  Aggregate v15 A1/A2 projections (`.80699/.81765`) hide this
fold failure; sealed v16 reaches only `.64172/.49549`.  Therefore the attention-15 dependency
factorial is currently ineligible.  Build and execute the separate A1/A2 oracle-axis plus exact
projective-bisector falsifier first.

Its prospective receipt is
`basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_construction_oracle_projective_bisector_v1.json`.
The reusable single-panel fit core is
`basis_aligned/bilinear_quotient/ops/construction_oracle_projector_fit.py`; it must be integrated
into a hash-bound managed runner without changing the completed parent artifact.

The integration is now complete in
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_projective_bisector_v1.py`
at reviewed SHA-256 `f1a1fae56a6c78e4a0b093ab49032649d8d1d66ecee3f2876540417f8f40bc3d`.
After confirming that hash and the current managed queue, enqueue it through `ops/enqueue.sh`; do
not run it directly.

Scope its v16 evidence as `OOD_TEXT_REUSE_NEW_INTERVENTION`: v16 text, native capability, and the
failed joint-projector response were already open, although the new oracle/bisector intervention
is frozen entirely from v15 before its v16 contexts are constructed.  It is intervention-held, not
a new pristine task discovery.

The bisector result has landed validly at SHA-256
`dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224` with terminal
`construction_conditioned_coordinate`.  Both construction oracles are stable/effective, their
same-head cosines are only `.409-.671`, and cross-use fails.  The bisector passes v15 target bars
but retains two P flips in one parity and fails v16 A2.  A naïve routed mixture is therefore not
selective because each own expert has the same parity-0 P flips.  The active successor is the
zero-forward CPU weight-convergence diagnostic in
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_construction_oracle_weight_convergence_v1.py`.

That weight screen has now returned `reader_equivalent_distinct_writes`: the construction axes do
not generally align as W_O writes or W_V pullbacks, but all five L15H5 Q/K/Q2/K2/V responses rank
top-ten for all four sources in both folds.  The next runner is
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py`
at reviewed SHA-256 `4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3`.
It must be enqueued through the managed GPU lane and interpreted before any L15H5-specific reset.

Do not overstate the exact-weight result.  A hash-bound zero-model path audit proved that every
current DAS outcome clamps the complete attention-15 donor head output after layer-15 Q/K/V has
been computed.  Thus the causal projector effect can travel through the residual skip/MLP15 and
layers 16-17, but cannot validate the weight-ranked `L15H5` Q/K/V reader path.  After the queued
construction result, run the frozen `upstream projector on/off x complete attention-15 on/off`
dependency factorial before any L15H5 reset/rescue claim.

That successor is now prospectively specified in
`basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_selected_projector_attention15_dependency_factorial_v1.json`,
with its derivation in
`basis_aligned/polynomial_causal/explanations/TEMPORAL_ISWAS_SELECTED_PROJECTOR_ATTENTION15_DEPENDENCY_FACTORIAL_DESIGN_2026-09-07.md`
and reusable CPU accounting in
`basis_aligned/bilinear_quotient/ops/two_by_two_dependency_contract.py`.  It is eligible only if
the multi-environment parent is mechanically valid and its registered joint-v15 feasibility
prediction passes.  Otherwise run the already committed projective-bisector falsifier first.

The target-feasible DAS run described below has completed.  It validly beats matched difference
in means (DIM) on cross-fitted A1 target transfer (`.8719` versus `.7250`) with stable rank-one
directions (minimum principal cosine `.8474`), so optimization is not the null.  It does not pass
construction-general identification: sealed A2 is `.6489`, below `.75`, and one P-control fold has
mean KL `.01945`, one flip, versus DIM `.00678`.  A complete 30-configuration red-team found that
the registered noise/Jacobian regularizers change selection scores by less than `.0005` and never
improve flips.  The current diagnosis is missing construction variation, not insufficient local
regularization strength.

The next multi-environment selector is already implemented and tested.  It keeps rank one fixed,
requires every fitted target construction to pass separately before control scoring, and selects
using worst P/C panels.  A history-disjoint third construction (`v16`: Right now/Back then and In
these/those days) is hash-bound in managed GPU lane 1 for a capability-only two-forward gate:

- runner:
  `basis_aligned/bilinear_quotient/ops/run_tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1.py`;
- reviewed SHA-256:
  `333c5398566539d8ba1f8940ab894d2ed8db9f1449caf73d5fd8d0156d40c34f`;
- queue order at this checkpoint: immediately behind live `v236`, before `v238` and `v247`.

Do not inspect a causal v16 outcome before the capability gate lands.  If it passes, score and
publish it, then use v15 A1/A2 as separate fitted environments and keep v16 sealed for the fixed
rank-one multi-construction causal transfer.  If it fails native capability, preserve the null and
do not repair the text post hoc.

An exact zero-forward weight translation of the learned four-head directions is also complete.
It maps each head coordinate through $W_O$ into residual space, ranks downstream Q/K/V and MLP
interfaces, and pulls it backward through $W_V^{\mathsf T}$ to rank earlier writers.  After a
preregistered scale-aware float32 audit, the diagnostic is valid: fold cosines are
`.9368-.9923`; every source ranks all five inspected `L15H5` interfaces in its top ten; and writer
pullbacks recover `L8H1 -> {L9H1,L9H4} -> L11H3` at `.9725-1.0` percentiles.  Treat this as an
explicit weight-compatibility hypothesis, not causal identification, until the sealed
construction transfer succeeds.

Latest Codex commits are `5f4acf140` (published weight-interface audit), `000d94082` (audit gate
repair), and `d1e461377` (audit preregistration).  The canonical dossier remains
`basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`.
The latest hourly clock is `HOURLY_STRATEGIC_REVIEW_2026-09-07_2215.md`; the next review is due at
the first safe boundary after 23:15 UTC.  The latest mathematical clock is
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2026.md`; the next is due after 23:26 UTC.

## Earlier circuit state at 2026-09-07 21:07 UTC

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

## Earlier continuation: target-feasible regularized DAS (completed; see restart delta)

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
