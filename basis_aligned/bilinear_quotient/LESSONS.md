# Lessons (distilled rules, each with a real example and a near non-example)

Hard-won rules of this program. Every lesson cites the section where it
was learned. A "near non-example" marks the boundary — where the rule
does NOT apply — because a rule without its boundary gets over-applied.

## 1. Mean-ablate; never zero-ablate (§353–354)
Zeroing is an off-manifold magnitude shock; it measures scale
sensitivity, not content.
- **Example:** zero-deleting ANY single head of a14 on IOI cost an
  identical ~1.64 margin (all nine within 0.005, recompute verified
  exact). Within-prompt MEAN ablation immediately differentiated:
  head 4 carries 0.93, most others ~0.
- **Near non-example:** whole-COMPONENT mean-ablation (the census's
  probe) is fine and always was — the rule is about deleting
  *sub-parts* (heads, direction bundles) whose absence shifts the
  parent's output scale.

## 2. Only subtraction bites; value-writing is void — with an offset tax (§352, §360–362)
Interventions act by removing variance along directions. The written
constant contributes nothing (optimized value == zero, +4.42 both;
natural donor values ≈ 0 effect) — except that off-manifold constants
(zero) pay ~1.6× extra vs the slice mean.
- **Example:** DAS "steering" of the line-break channel: +4.42 with
  the optimized value, +4.42 with all-zeros, +0.16 with natural donor
  values. The optimizer had learned *which directions to delete*.
- **Near non-example:** r.6.0.0 (agreement 0.32 across constants) is
  the one leaf where constants disagree beyond the offset tax —
  earmarked; the law is a strong default, not yet a theorem.

## 3. Selection effects: census members are damage-TAILS (§net_utility, layout-brake page)
A leaf's members are the positions where machinery moves loss MOST —
including where it's most wrong. Member-mean damage can be negative
while the circuit is corpus-wide useful.
- **Example:** r.3.1.0: members −1.14 mean (ablation "helps") but
  corpus-wide +0.071/token (+3,840 nats) — optimality holds.
- **Near non-example:** if a leaf's GLOBAL net were ≤0, "the model
  would be better off without it on this corpus" is a fair claim —
  that's the slack-harvest regime, and it did improve the model
  (−0.048 fresh) when sign-calibrated.

## 4. Richer feature sets overfit fixed-data greedy search (§347, §357)
At fixed data, adding features can LOWER held-out performance.
- **Example:** quadratic 1680-dim probe: AUC 0.551 vs plain ridge's
  0.621. Exact pair-fold features: converged 55 vs 58 without them.
- **Near non-example:** adding the 10 class labels lifted program
  passes 2/16 → 5/16 — a few *high-prior* features help; the failure
  mode is bulk vocabulary, not features per se.

## 5. Two-signed policies are universal; stories must state both wings (§348–349)
100% of tested leaves are sign-mixed. "Helps X" alone is an incomplete
story by construction.
- **Example:** line-break circuit = mlp0 push MINUS mlp3 brake; the
  Westminster-Abbey token improves 6.8 nats when the wrong-there push
  is removed.
- **Near non-example:** wings are tails of ONE continuous mode, not
  two modules — don't reify "the positive circuit" and "the negative
  circuit" without a bundle dissociation test (§349's (a)).

## 6. Shape and gain are different objects in attention (§364)
Unnormalized squared-product patterns = shape × gain. Linear fits on
raw patterns measure gain variance.
- **Example:** archetype dictionary R² ≥0.7 for 1/162 heads while
  literal one-hot pattern swaps work for 71/74 — both true, different
  objects.
- **Near non-example:** functional claims ("this head reads prev")
  survive; matrix-level claims ("this pattern ≈ prev matrix") don't.

## 7. Tags are tree-instance-local (§344 caveats)
The same physical circuit gets different tags in different tree
builds. Identity = member overlap, never name.
- **Example:** mb3's r.0.0.1 ("rare multi-token words") vs the
  212-row tree's r.0.0.1 (line-break) — different circuits, same tag.
- **Near non-example:** within ONE census_state.pt, tags are stable
  and safe to use as keys.

## 8. Description-language limits are measured, not assumed (§341, §346, §357, §363)
The programmable frontier (55–58/118) is invariant to input-feature
vocabulary: surface, classes, triggers, shifts, folds, pair folds,
unbounded match transports. Membership for the rest is not
token-definable.
- **Example:** five feature classes, same plateau ±3.
- **Near non-example:** the plateau is about INPUT-computable
  features; stream-level features are a different class (probe: 3.8×
  vs surface 1.5×) and are tested separately.

## 9. "Mechanism" has grades; name them honestly (§365-pending, user standard)
Firing-condition mechanisms (trigger tables, 8–36× precision) are NOT
computational mechanisms (executable code replicating behavior).
- **Example:** the four "induction-grade" circuits are trigger-grade;
  zero circuits currently meet the write-the-code-and-replicate
  standard (first attempts queued: mech_replicate, suffix_code).
- **Near non-example:** the assembly's fold tables DO replicate
  behavior as code — at component grain; the missing piece is
  circuit-grain semantics, not code per se.

## 10. Say which distribution a "fresh" number used (§386)
The model trained on FineWeb; the program's fresh legs used Pile --
mildly OOD (+0.10 base CE). Transfer that survives OOD is extra
safe, but absolute numbers mix a distribution tax into the
replacement tax.
- **Example:** combined-readable +0.224 "fresh" = Pile; the
  in-distribution number is expected slightly lower.
- **Near non-example:** internal comparisons (config A vs B on the
  same rows) are unaffected -- the tax cancels.

## Ops rules (each cost at least one incident)
- THE BOX IS NOT VOLUME-BACKED (`workspace_is_volume: false`). A recycle on
  2026-08-24 wiped the venv, the HF cache, /workspace/rspd and the bqrunner
  service; only git survived, and only because everything was pushed. Two
  standing consequences: (1) push after every writeup, no exceptions; (2) the
  rebuild is scripted — `bash ops/restore.sh` — so a recycle costs ~5 minutes
  instead of a session. Anything you set up by hand OUTSIDE the repo (a
  supervisor service, a cloned dependency) must be committed under ops/ the
  same hour you create it, or it is not real.
- NO HF_TOKEN IS SET ON THIS BOX, and it costs real throughput. `vast-capabilities`
  reports `credentials.huggingface: false`; there is no env var, no
  ~/.cache/huggingface/token, and nothing in ${WORKSPACE}/.env. 297 runlogs carry
  "You are sending unauthenticated requests to the HF Hub... set a HF_TOKEN to
  enable higher rate limits and faster downloads". MEASURED COST on 2026-08-27:
  `writer_floor_question` spent >14 min caching 3x96 FineWeb rows (normally 2-4
  min) at ~11% CPU, network-bound, while lane 2 streamed concurrently; restore.sh
  lost the sqrd12 config.json to a mid-transfer server disconnect; channel_depth
  hit SSL-EOF retries. Two lanes streaming FineWeb at once is the worst case.
  FIX (needs a human, one line): `echo 'HF_TOKEN=hf_...' >> ${WORKSPACE}/.env`
  then restart the runners. Until then, expect row caching to dominate short runs
  and do not diagnose a slow FineWeb load as a hang -- check CPU-tick progress
  before killing anything (a stalled lane and a network-bound lane look identical).
- queue.txt takes ABSOLUTE paths; bare names are silently dropped.
- ...and so does every shell command an agent runs. The cwd does NOT persist
  reliably between tool calls: a `cd BQ && ...` that runs from BQ FAILS (already
  there), and with `&&` chaining it silently skips everything after it. On
  2026-08-27 that dropped a script's registered-predictions header (caught before
  queueing) and produced three separate false alarms -- a "missing" ledger
  section, a "hung" run, a "missing" runner.log -- each of which was a relative
  path resolving from the wrong directory. Use absolute paths, or set
  `BQ=/workspace/.../bilinear_quotient` at the top of every command.
- After writing a transform-generated script: verify file exists AND
  ast.parse it BEFORE queueing (ioi_chain incident: queued, then the
  transform crashed, runner would have skipped silently).
- Watchers: count-based on _completed.txt, never grep of logs (stale
  tracebacks false-trigger).
- git: explicit paths during concurrent work; agents never commit
  (consolidator model); directory-wide `git add` swept another
  agent's uncommitted work into a pushed commit once (§356).
- Registry/features writes only through census_lib (locks, deep-merge,
  append-only certification).
- Mid-wave infra edits cause version skew inside running agents:
  batch infra changes between waves; provenance records lib_rev.
- bqrunner REQUIRES ABSOLUTE PATHS in queue.txt: it checks `[ -f "$line" ]`
  from the runner's own cwd, so a relative path (e.g. `weight_action_compose.py`)
  is popped and SILENTLY DROPPED -- nothing runs, no error. Always queue the full
  /workspace/.../script.py path (§753->754 cost one idle cycle to this).
- CORRELATION BIAS CONFOUND (751/752/754): a corr between two model-derived
  vectors that share a large mean/bias rides that shared component -> real ~=
  shuffled. ALWAYS center (subtract per-dim mean over samples) before correlating,
  and gate on a shuffle/permutation null. Cost 754 a wrong 0.378 (true 0.026).
- FOLD names GEOMETRY, not causal FUNCTION (792): folding the QK content bilinear
  form onto class directions gives the QK class-attention GEOMETRY, but whole-head
  ablation shows the head contributes BROADLY (multi-function value-moving swamps any
  class sub-circuit). Naming "head X = concept Y" needs EDGE-level (specific coupling)
  ablation, not whole-head; and the content fold ignores rmsnorm/rotary/value-path.
  Robust framing: which VARIABLE a head READS (input-restriction 784), not per-head
  concept labels.

## Centered keep-only silently drops the per-component MEAN (DC bias) — §804/805
The keep-only / CE-recovery metric substitutes proj_U(v), with U built from CENTERED
data (token-conditional means, own-SVD of centered output). This discards the
per-component MEAN — the constant bias the component adds at every position. For most
components the mean is not loss-critical, so keep works. But when a component's output
is dominated by a large constant bias (gpt2-medium mlp0: mean = 91% of output norm),
dropping it makes keep spuriously NEGATIVE (worse than ablation) at EVERY rank — even
keeping the output's own top-128 SVD directions. This masqueraded as "genuine
non-separable computation" (§802) for a whole wake before gpt2med_dc_test caught it.
RULE: before calling a keep-only score negative or a component "not low-rank", check
mean-norm / output-norm; if it's large, redo keep with the mean PRESERVED
(v_kept = mean + proj_U(v − mean)). A large constant MEAN is NOT a per-token massive
activation (norm max/mean can still be ~1.5) — they are different diagnostics.
- EDIT-BEFORE-QUEUE (§1349): a path in queue.txt is LIVE the moment it is echoed —
  bqrunner popped exclaim_gates.py between generation and an in-place rewrite, running
  the version already diagnosed as ill-posed. All edits and the ast.parse check complete
  BEFORE the echo into queue.txt, never after. (Related: asserted transforms, §1347.)
- JITTER CONTROL SCOPE (§1362): jitter ("wrong position, same text") is valid only for
  POSITION-SPIKED capabilities. For phrase-scoped services (a8's numeric-context), jitter
  positions sit inside the service's own support and the control is not evaluable — use a
  matched-phrase-elsewhere draw instead, and say which regime the screen uses. (Two
  contamination classes + one refuted account: §1339 mask contamination, §1361-62.)
- PATCH TEMPLATES, NOT MEMORIES (§1376): three bar-keying failures (§1348, §1354, §1376)
  shared one root — a bar-design lesson recorded in prose while the template file that
  embodies the old design kept being copied. When a design lesson lands, patch every
  template script in the same commit; generated screens must report BOTH raw winner and
  best specificity-passing candidate with bars keyed to the latter.
- Existence-check subtype masks against the corpus BEFORE registering ratio predictions on them (S1416: nl_para/nl_mixed n=0 made pred_b vacuous — the corpus tokenizes newlines only as bare '\n').
- Path-patching a residual contribution requires an EXACT running ledger under lambda-mixing: rescale every stored contribution by lambda0 each layer, credit lambda1*x0 to the embedding, and ASSERT sum(contrib)==x on the first batch BEFORE patching (S1426: unscaled ledger over-subtracted 30x at depth; the docstring even had the caveat and reasoned it away).
- Terminology (user, 2026-08-25): say "inputs"/"reads", not "diet"; define any abbreviation (PC = principal component) at first use in writeups. Variance-PCs are a computational convenience, NOT a claim about a module's natural features (S1425/1429: two basis families lost to random — the natural basis question is open and belongs to each module's OWN weight structure).
- Anchors are FROZEN inputs, never recomputed inside experiment scripts (S1438: in-script "mean" arm computed after residualization = near-zero anchor, silently wrong; the sweep's frozen ce_mean/ce_opt on identical rows+mask caught it). This is TheseusBench Invariant 4, learned by stepping on it.

## 11. Consult the repo's record before acting on your model of the situation (2026-08-27 session)
A fresh session's first instinct is to *build*. This repo has already solved
most of what a fresh session wants to build, and the cost of checking is one
grep. Three failures in one hour traced to the same root: producing output
before consulting the record.
- **Example:** after a recycle, the driver wrote `bootstrap.sh` (duplicating
  `ops/restore.sh`), `msg.sh` + `watch_channel.sh` (duplicating
  `AGENT_BOARD.md`), and nearly committed a SECOND message channel — which
  would have split Claude<->Codex traffic so that each agent saw half the
  messages. `SWARM_RUNBOOK.md §0` is literally a new-session bootstrap
  checklist and was found ~40 min in, after the redundant work. Also in the
  same hour: told Codex its oracle arm was probably circular (checking §1515
  showed it was not — greedy explicitly BEAT the weights-only top-5), and
  flagged a falsifiability risk in Codex's registered bar that a 40-second
  weights-only measurement later refuted (half-oracle .404 vs random p95 .117).
- **Near non-example:** the tmux two-agent session (`tools/dev-session.sh`) was
  genuinely absent and worth adding — the rule is "check first", not "never
  build". The check is what distinguishes the two cases, and it is cheap.
- **Standing form:** before writing any script, tool, or doc, grep
  SWARM_RUNBOOK / LESSONS / ops/ / BENCHMARK_BACKLOG for the thing you are
  about to create. Before flagging a concern to another agent, ask whether it
  is measurable in under two minutes — if so, measure it and report the number
  instead of the worry. A retracted flag costs the other agent a design detour.

## 12. A watcher's own failure modes are the ones nobody watches (2026-08-27 session)
Monitors get written for the *content* of their filters and shipped without
testing their *lifecycle*. Both failure directions cost real time in one hour.
- **Example (too loud):** `watch_runs.sh` announced every pre-existing log in
  `runlogs/` as "new" — ~1000 events on a freshly cloned repo, which
  rate-limited the monitor and buried the real signal. Fix: seed all existing
  files SILENTLY at startup; only post-startup files are events.
- **Example (too quiet, worse):** `watch_board.sh` used `git pull --rebase`,
  which FAILS whenever the bqrunner lanes have result JSONs and runlogs dirty
  — i.e. exactly whenever experiments are running and Codex is most likely to
  post. On failure it read a stale local file and went silent, which is
  indistinguishable from "nothing posted". Fix: `git fetch` +
  `git show origin/main:AGENT_BOARD.md`, which never touches the working tree.
- **Near non-example:** a watcher that is merely *chatty* about real events is
  fine; the failure is announcing NON-events (startup backlog) or swallowing
  real ones. Test both explicitly before arming: run it against the existing
  state (does it flood?) and against the degraded state — dirty tree, missing
  file, no network (does it go quiet when it should speak?).

## 13. Never take a fixed absolute tolerance to a spectrum (§1601 era, 2026-08-27)
Eigenvalue "zeros" sit at the data's noise floor, which scales with the
spectrum. A constant threshold is wrong at every scale but one. This trap fired
TWICE in one hour, independently, in two agents' code.
- **Example (driver):** checking Codex's inertia bound, `tol=1e-8` on a rank-2
  matrix with eigs (+144.9, −73.8) in float32 reported inertia **(6,5)** — 
  float32 noise at scale ~145 is ~1e-3, five orders above the threshold.
- **Example (Codex):** `scalar_quadratic_bilinear_factors` derives
  `tolerance = finfo(matrix.dtype).eps * max(shape) * scale` AFTER
  `matrix.double()`, so eps is always float64's 2.2e-16 — read off the STORAGE
  dtype, never the data's actual precision. A form computed from bilin18's
  float32 weights prices at **576 gates instead of 1**, and fails silently: the
  returned factorization still reconstructs to ~1e-13.
- **Rule:** scale the tolerance by `max|eig|` AND by the precision the data
  actually carries (the dtype it was COMPUTED in, not the one it is stored in).
  Upcasting does not clean float32 noise. Test any spectral code on a dense,
  float32-derived matrix at real width (D=1152), never only on exact diagonals.

## 14. A nonzero exit does not mean a failed experiment — check for the result file (§1602)
Scripts that stream FineWeb can abort during interpreter *finalization*, long
after the science is done and written. `_completed.txt` records the signal, so
the run looks failed to every watcher and to the next wake.
- **Example:** `channel_depth` exited **134** (SIGABRT) with
  `PyGILState_Release: thread state must be current` during shutdown, from the
  HF datasets streaming thread that had retried an SSL EOF mid-run. Every result
  was written and exact (reconstruction rel err 6.3e-08). A wake that trusted
  `exit=134` would have re-queued 491 s of completed work.
- **Near non-example:** a nonzero exit with NO result JSON, or one whose JSON is
  short of the registered keys, is a real failure — do not rationalise those.
  The test is the artifact, not the exit code.
- **Rule:** on any nonzero exit, check whether the results file exists and
  contains every registered prediction key BEFORE requeueing. Mitigation for new
  scripts that stream data: `sys.stdout.flush(); os._exit(0)` after the final
  write, which skips the finalizer that crashes.

## 15. A random subspace has "writers" — control for the high-norm floor (§1606)
Component attribution onto ANY low-rank subspace is dominated by the components
with the largest output norm, whether or not the subspace means anything.
- **Example:** a random orthonormal rank-8 basis at pronouns@mlp17 returns
  consensus top-6 writers mlp17, mlp16, mlp15, attn16, attn17, attn13. All three
  real slice rules (|λ|-top, positive payload, negative gate) share exactly
  {mlp17, mlp16, mlp15} with it. Raw Jaccard(abs, pos) = .333; floor-corrected it
  is **.000** — the entire apparent agreement between two rules was the floor.
  Two of §1598's six reported writers are floor components.
- **Near non-example:** WITHIN-layer relative measures (head grain: which of the
  9 heads dominates its layer, at what ratio) are not obviously exposed, because a
  global norm floor largely cancels in a within-layer ratio. That is a hypothesis,
  not a result — it needs its own control before being relied on.
- **Rule:** every component-level writer/attribution claim carries a
  random-subspace control at matched rank, and only the floor-corrected set is
  named. Report both the raw and the corrected overlap; the difference between
  them is often the whole result.
- **EXTENSION (§1612): report the random SHARE too, not just the random top-k —
  and match the published statistic exactly.** The null share is CELL-DEPENDENT:
  .4489 at question@mlp11 (rank-2, TOP=4) but .7295 at pronouns@mlp17 (rank-8,
  TOP=6). A certified slice sat ABOVE its null at the first cell (.5563) and FAR
  BELOW at the second (.5846), so a bare share cannot be read as evidence of
  either concentration or diffuseness. The informative quantity is always
  **share minus null**.
- **And the currency must match.** §1609-§1611 measured POSITIVE-ONLY signed
  share while the published figures use ABSOLUTE attribution mass
  (`slice_writers.py:216`, `slice_writers_p.py:205`). That mismatch did not just
  weaken a conclusion, it INVERTED one: §1610 reported attn10 as
  indistinguishable from floor, and under the published currency attn10 is absent
  from the random top-4 in 3/3 and is rule-specific. **Read the source of any
  published number before testing it; a control in the wrong currency can impugn
  a result that was fine.**

## 16. Validate serialisation BEFORE the expensive compute, not after (§1608 era)
`json.dump` runs last, so a non-serialisable value in the results dict destroys
the entire run's artifact after all the GPU work is done.
- **Example:** `headgrain_control2` ran all 12 depth-curve passes (~9 GPU-min),
  then died at `json.dump` with `TypeError: Object of type set is not JSON
  serializable` — a `certified` roster stored as a `set` in the config block. The
  results file was left 0 bytes. The measurements survived only because every
  number had been `print(..., flush=True)`-ed as it was computed.
- **Near non-example:** this is NOT the §1602 finalizer crash (LESSONS 14), where
  the artifact was written correctly and only the interpreter teardown failed.
  The test distinguishing them is the artifact: 0 bytes or missing registered
  keys = real failure; complete file = benign.
- **Rules:** (1) print every headline number with `flush=True` as it is computed,
  so a serialisation failure costs the artifact but never the science;
  (2) pass `default=lambda o: sorted(o) if isinstance(o, set) else str(o)` to
  `json.dump` so no future stray type can cost a run; (3) if a results dict
  contains config echoed from module constants, dump-test it on CPU before
  queueing — it costs milliseconds.
