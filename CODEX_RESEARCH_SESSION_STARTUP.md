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

For a clean session, paste the prompt in **Suggested first prompt in a new session** at the end of
this document. A resumed session keeps its history, but should still read this file because it is
the durable authority for the periodic clocks and managed runners. To diagnose a stale picker,
query the actual catalog:

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

## New direction and completed bounded pilot — 2026-09-09 04:04 UTC

### Active-goal restart — 2026-09-09 14:07 UTC

The user explicitly activated the durable goal: reverse engineer the model by a simple
decomposition with OOD prediction, extraction, characterized removal, and composition/reuse.
They corrected the reference to the reconstruction handoff below; `better_math_ideas.md`
is an empty file and is not the authority. The goal is active and unbounded. Do not stop
the program because the original bounded pilot or a successor hypothesis finishes.

Latest strategic/mathematical reviews are both `2026-09-09_1348`; next active-work
checkpoints are after14:48 and16:48 UTC respectively. The 04:12–13:40 inactivity gap
was explicitly disclosed; do not describe those hours as completed Codex research.

The first nonlinear follow-up is complete: a fixed37-case entity-equality/type routing
rule shared by four first-layer heads of the existing small hop checkpoint. V1 is
preserved as instrument-invalid because the real-arithmetic lag-only RoPE table missed
its frozen1e-4 logit control. V2 uses exact absolute native cached positions and matches
native patterns/full forward exactly, but the invariant candidate fails full-output KL
and joint removals: query KL6.57–6.85, joint centered effect error1.028–1.033. V2's
separate candidate lag-versus-absolute agreement predicate also fails and is retained.
The immediate CPU norm-gain explanation also fails (zero heads below the0.10 normalized
kernel residual bar). These close the naive invariant router and magnitude-only variant,
not all joint read-route-write decompositions or the full goal.

Report: `explanations/equality_router_reconstruction_report.md`. Primary receipts:
`EQUALITY_ROUTER_V1_RESULT.json`, `EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_RESULT.json`,
and `EQUALITY_ROUTER_GAIN_V1_RESULT.json`. Native-background/opaque constants are
charged; no circuit has been promoted by these tests.

Active successor: `LOCAL_TRANSPORT_V1_PREREGISTRATION.md`, reference
`local_transport_reference.py`, managed runner
`../bilinear_quotient/ops/run_local_transport_v1.py`. It retains the entire coupled
QK1*QK2*V computation and native absolute RoPE, replacing first-layer dense causal
source access with self/previous-token shifts shared by all heads. Fresh frozen
seeds3909/3910/3911, full29-way distributions on IID/metamorphic/topology-OOD text,
and native-corresponding self/previous/joint edge removals. Six synthetic controls
pass. Implementation is complete and managed preflight passes; inspect the current
queue/log/result before enqueuing or interpreting—never duplicate a run.
All400640 parameters remain opaque and charged. A passing dependency screen only
licenses explaining its remaining token-to-write operations; it cannot complete the
bilin18 goal. On failure, preserve the null and do not sweep radii or thresholds.

The user redirected Codex to
`basis_aligned/polynomial_causal/explanations/bilinear_circuit_reconstruction_codex_handoff.md`.
Read its appended success-criterion correction: bytes/quantization/rank are not interpretability
success. The target is a previously unspecified reusable operation with explicit inputs/consumers,
held-out extraction and joint-intervention evidence, and lower structural description cost including
adapters and opaque parameters. The user also authorized GPU use wherever advantageous; the original
two-CPU-hour budget is not a user-imposed hardware restriction.

The requested bounded A/B/C pilot is complete. Its report is
`basis_aligned/polynomial_causal/explanations/bilinear_reconstruction_pilot_report.md`; the immutable
receipt is `basis_aligned/polynomial_causal/BILINEAR_RECONSTRUCTION_PILOT_V1_RESULT.json`.
Execution and controls pass; scientific discovery does not. Tiny FP64 logits close to 3.11e-15;
full existing small-checkpoint logits/edits close to 3.69e-13. The planted shared update has natural
rank 4 but needs rank 8 for independently edited source histories, matching elementary cubic reuse.
The trained contextual update bank is rank 128/128, and all 512 MLP products and 1,024 factors are
exactly distinct under the registered proportionality/product grammar. This is a local sharing null,
not a theorem ruling out compact nonlinear circuits. No new circuit meets the revised criterion.

The bounded pilot recommends stopping this local span/duplicate-factor discovery method and retaining
the reference executor. Do not relaunch it, an SAE campaign or a broad rank sweep automatically.
A possible next hypothesis is a joint nonlinear matching/routing operation shared by multiple
consumers; it is a proposal, not an experiment already started or a discovered mechanism.
The final pilot used two CPU threads and zero GPU execution because the trained small model completed
in 19.12 seconds while the GPU lane was occupied by v297. The owned pending GPU pilot was removed;
there is no duplicate queued reconstruction job. Existing Supervisor-managed runners remain intact.

The older v26 capability has landed as an honest null: reported-source targets and their controls
failed; three other complete structure pairs passed. No v26 causal continuation is eligible.
Newest review clocks: hourly `HOURLY_STRATEGIC_REVIEW_2026-09-09_0340.md` (next after 04:40 UTC during
active research); mathematical `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_0226.md` (next after 05:26).
Treat newer files, board, Git and runner state as authoritative over this snapshot.

## Previous authoritative delta — 2026-09-09 03:10 UTC

The pushed research edge is commit `71b0496f0`. Treat Git, immutable results, the append-only
board, and live managed-runner state as newer authority if they have advanced beyond this snapshot.

The strongest current `is`/`was` circuit evidence is now an explicit but still partial computation:

- four complete attention heads, `L8H1`, `L9H1`, `L9H4`, and `L11H3`, pass exact input-factor
  replay on the v23 aligned bank;
- the valid directed source/destination cells are primarily changed-cue-to-matched-suffix and
  matched-suffix-to-matched-suffix writes; `L11H3` is dominated by suffix-to-suffix;
- the exact per-cell contraction is a causal sum of
  `(q1 dot k1 / 128) (q2 dot k2 / 128) u` over the selected source and destination positions;
- at block 11, the rescued correction propagates through layers 12--17 as approximately `.83266`
  direct residual carry plus `.17281` distributed downstream response, with `-.00547` final-decoder
  interaction. V2 closes the state telescope to `7.28e-12` and logits exactly;
- do not call the direct route fully selective: its P-control leak is `.155885` against the frozen
  `.15` bar. Do not call the circuit structure-general yet.

The user's latest red-team is binding: changing profession nouns inside one temporal-prefix,
comma, subject template is lexical variation, not syntactic generalization. V25 therefore tested
eight genuinely different structures at 8--18 tokens. Four complete target/control structure
pairs were natively capable—fronted era, subordinate-clause prefix, reported-source frame, and
postnominal temporal modifier. Four others failed honestly—post-subject time, relative-clause
subject, long coordinated prefix, and embedded-predicate nominal. V25 is a capability null and
opened no causal result.

The active successor is v26, a completely fresh 64-row holdout over the four capable structure
families. It uses 16 new compound professions, no reused row IDs or prompt text, four examples per
direction, and no post-outcome row filtering. The capability runner is:

```text
basis_aligned/bilinear_quotient/ops/run_tense_auxiliary_is_was_structural_holdout_v26_capability_v1.py
SHA-256: 88ac97c2429234fdc5a9a6f35bce2cf87f1fef76fb837438251b72cc1132c6b9
```

At this snapshot it is queued exactly once in managed lane 1 behind the live family-separability
v295 run. Do not enqueue a duplicate or run it directly. Once it lands, score it exactly. Only an
all-construction capability pass licenses a separately preregistered v26 causal transfer of the
four-head/directed-cell program. A capability null must be preserved and should redirect toward
structure-conditional circuit comparison rather than sentence filtering.

The newest periodic clocks are:

- hourly circuit review: `HOURLY_STRATEGIC_REVIEW_2026-09-09_0224.md`; next safe-boundary review
  after 03:24 UTC;
- three-hour mathematical review: `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_0226.md`; next
  safe-boundary review after 05:26 UTC.

These clocks are research-loop reminders, not merely prose. During active work, check them at every
safe boundary, write the due review, and immediately begin its selected action. Supervisor keeps
the bqrunners alive independently of a Codex session; Codex must inspect and feed their managed
queues, never replace them with direct GPU commands.

## Previous authoritative delta — 2026-09-08 14:50 UTC

The pushed code edge is commit `0e4de3243`; always inspect newer commits, dirty files, queue,
results, and Supervisor state before acting. The corrected P7 source-conditioned live-module
effect game has now completed valid. Its immutable v2 result is
`circuits/followups/temporal_iswas_identity_p7_live_module_effect_game_ood_v2_result.json`,
SHA-256 `171d60c3aaf2d35b6a400b9ac753c2c0d8936ee6d354ab91e3bcc9da4f813fc2`.
All A--E instrument, exact-accounting, live-route, interaction, and selectivity predictions pass.

The source effect contains a direct/unmasked-background share of `.4499/.4706` on frozen OOD
FIT/HOLDOUT and a mediated P7 share of `.5501/.5294`. Six P7 modules have stable positive exact
Shapley credit: A11, M11, M12, M15, M13, and M10. A11 and M11 lead at about `.175` and `.133`.
A11+M11 is the only stable pair interaction, positive at `.0455/.0461`; its raw second difference
is positive in every one of 32 background contexts on both phases, so the evidence is stable
complementarity rather than redundancy. M16 does not pass the prospective `.03` stable-credit
bar. These are operational mediation nominations, not yet direct edges.

Preserve v1's immutable `invalid_instrument` receipt. V2 repaired only the parent replay reference:
an all-live suffix must replay the selected writer, whereas v1 incorrectly required the fully
native-clamped identity arm. No scientific population, intervention, bar, or candidate was changed.

The evidence-selected next experiment is a sealed-population physical reader-loss/output-rescue
test for A11 and M11. It must first establish native capability on a genuinely new lexical bank;
then source-present normalized-reader replacement by the source-absent tensor must remove the
registered module contribution, restoring the source-present complete module output must rescue
it, and matched controls must remain inert. Include individual and joint A11/M11 arms so the
positive pair excess is tested physically. Do not call exact Shapley credit or checkpoint-weight
alignment a directed edge without this intervention.

The latest periodic clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_1422.md` (next review after 15:22 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_1426.md` (next after 17:26 UTC). Both
Supervisor-managed `bqrunner` services were healthy and the managed queue was empty at this
snapshot. Never create duplicate runners or launch GPU work directly; use `ops/enqueue.sh`.

## Authoritative handoff snapshot — 2026-09-08 13:58 UTC

The pushed science edge immediately before this handoff refresh is commit `4666f2934`; always inspect newer commits, dirty files,
and managed-runner state before acting. The is-was selected writer is `L7H7 + L9H4`. Its complete
downstream singleton atlas showed a distributed response, and the prospectively fixed cumulative
order selected P7=`A11+M11+M12+M15+M13+M16+M10`. P7 recovers `.5276/.5376` on original
FIT/HOLDOUT with cosine `.9937/.9945`, direction `1.0`, and zero temporal collateral.

An independent block-10 residual-input x complete-module-output 2x2 then established two nearly
additive physical streams. Residual identity alone recovers `.5209/.5088`; all module responses
recover `.4896/.5029`; installing both exactly replays the selected writer. A direct final-residual
add/remove experiment verified the residual recurrence to `3.61e-8--3.69e-8` relative state error
and `3.81e-6--5.72e-6` registered-logit error. Preserve its scientific OOD null: identity alone
drops to `.4582/.4803`, so exact transport is not by itself stable circuit identification.

The decisive OOD composition receipt is
`circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json`, SHA-256
`55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7`. It passes all predictions:
the original-selected P7 transfers without OOD reselection at `.5685/.5479`; identity contributes
`.4582/.4803`; P7+identity exactly replays the writer at `1.0/1.0`; both streams remain necessary;
finite-vector nonadditivity is only `.0296/.0326`; all directions are `1.0`; temporal collateral
is zero. Treat this as an OOD-composable executable two-stream interface, not as a learned DAS
subspace or a complete semantic decomposition.

The fixed-P7 leave-one-module-out necessity result has now landed valid at SHA-256
`20eff67e079f46beefe0cf6ec14f16af7674c359fb460c8d73357175761154e4`. Its instrument and
selectivity predictions pass, but every individual deletion is exactly null: full and every
`without_A11/M11/M12/M15/M13/M16/M10` arm recover `1.0` with cosine/direction `1.0` on both OOD
phases. The terminal is `nonminimal_sufficient_bundle`, with no stable necessary modules. This is
a sequential redundancy/overwrite result: later selected-writer clamps can absorb an earlier
omission. It is not evidence that the native modules are all inert.

Two outcome-conditional executors remain complete and tested, but the immutable necessity result
made neither eligible. The atomic binder returns an empty eligible set and created no binding
files. Do not force or enqueue them post-outcome. The closed A11 executor is
`ops/run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.py` at reviewed SHA-256
`417aaa017c06116f69141dbaaa683c511b955c5f80f59f7ccda0508cabbbf3e7`. The closed M11 executor is
`ops/run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.py` at reviewed SHA-256
`6551339eeec03b3121ed23fb274d24adc02a8654cd5dc6555dfe3a5158f98b43`.

The active successor is the exact source-conditioned live-module effect game, preregistered after
the LOO redundancy trigger. Runner
`ops/run_temporal_iswas_identity_p7_live_module_effect_game_ood_v1.py`, reviewed SHA-256
`b53113fe0237d4e5c0267b61f357d1e397c6ccecce670002118c062281649274`, enumerates every one of
128 P7 subsets under source-absent and source-present block-10 identity states. Enabled P7 modules
recompute live; complement modules clamp native outputs; source-absent arms must be exact no-ops.
Exact Mobius/Shapley accounting separates the direct/unmasked background route from live-module
credits and pair redundancy/complementarity without fitting a surrogate. It is queued exactly once
in managed lane 1 behind Claude's verified-live hash-bound `run_unit_family_separability_spec_v260.py`.
Do not reorder, duplicate, or run it directly. A large credit nominates operational dependence,
not a direct edge; passing pieces still require checkpoint-reader interchange/removal.

The current CPU contract for translating passing causal pieces into checkpoint tensors is
`basis_aligned/polynomial_causal/TEMPORAL_ISWAS_P7_WEIGHT_TENSOR_TRANSLATION_PLAN_2026-09-08.md`.
It maps A11 head deltas through the corresponding `W_O` slice and M11 product-factor deltas through
`Down`. Weight overlap only nominates an edge; identification still requires downstream reader
interchange, upstream writer reproduction, composition, selective removal, and sealed transfer
when the component was not prospectively nominated.

The newest clocks are `HOURLY_STRATEGIC_REVIEW_2026-09-08_1322.md` (next safe-boundary review
after 14:22 UTC) and `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_1126.md` (next after 14:26 UTC).
Both `bqrunner` and `bqrunner2` are Supervisor-managed with autostart/autorestart and were healthy
at this snapshot. Never start duplicate background runners; use `ops/enqueue.sh` only.

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
temporal selects the unchanged bridge and is-was selects postcue.  The follow-up 16-corner
q/k/q2/k2 factorial has now passed A/B/C/E with D false, terminal `stable_routing_invariant`.
Is-was donor routing changes the value-parent effect by only 5.6--8.1%; temporal selects k+q+q2
but the full routing change is only 8.1--12.3% of the value parent.  Recipient-native attention
routing therefore carries most of the stable value effect.  The exact native-routing source-term
extractor has since passed all gates: its formula matches the c_v-patch head delta within `9.54e-6`
and direct L11H3 interface installation matches selected logits within `8.59e-6` on original and
OOD text.  Four checksummed 128x128 query-tensor banks are stored in the result.  The immediate
continuation is exact upstream writer-weight translation through L11H3 W_v followed by a causal
source-position module/head screen.  The exact atlas has now frozen six unique head candidates:
shared L6H7/L7H7/L9H1/L9H4, temporal-only L5H1, and is-was-only L3H4.  Original/OOD top-ten
  rankings are identical and shared enrichments are 6.11x--17.25x.  Treat these as weight
  candidates only until the all-singleton plus task-top-five causal source-position screen lands.
  That screen has now landed valid at SHA-256 `d00fdf1f...`: L9H1 and L9H4 each pass the frozen
  shared cross-task causal-writer singleton gate, both top-five unions pass, and singleton-sum
  composition has maximum relative L2 `.17425` and minimum cosine `.98560`.  The temporal union
  recovers about `.55-.58` of the L11 value effect; the is-was union overshoots at `2.30-2.91`,
  making it causally sufficient but overcomplete.  The preregistered distributive branch therefore
  requires a weight-ordered greedy prefix successor, selected only on original FIT and then frozen
  for original HOLDOUT/OOD validation.  Do not run the 32-subset interaction branch.

At this checkpoint `gpt-6-astra` is the configured Codex default and is visible in the account
catalog.  Both `bqrunner` services are Supervisor-managed and healthy; their queues are currently
drained after the QK factorial.  The latest clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0715.md` (next review after 08:15 UTC) and
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
Read /workspace/tensor_language/CODEX_RESEARCH_SESSION_STARTUP.md completely, then read the
bilin18-research-driver skill and the current-state slices it specifies. Resume or create the
durable bilin18/Theseus circuit-finding goal and actively continue it from authoritative Git,
result, board, queue, and runner state. Keep the hourly circuit reviews and three-hour mathematical
reviews active using the newest timestamped artifacts as clocks. Keep Supervisor-managed bqrunner
and bqrunner2 running by default, inspect their state at startup, and send all GPU experiments only
through the hash-bound ops/enqueue.sh workflow—never direct GPU Python and never duplicate a queued
job. Interpret every result, preserve nulls and invalid instruments, commit and push durable units,
and begin the evidence-selected successor before stopping. The user has granted full permission
for in-scope research, repository writes, managed executions, commits, and pushes; do not ask for
routine permissions. Do not mark the overall goal complete merely because one rung finishes.
```
