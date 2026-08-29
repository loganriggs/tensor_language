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

## 17. On ~10 units, p < .05 carries almost no information — confirm on a DISJOINT set before quoting (§1614-§1616)
Two correlations that looked solid on ten classes evaporated on ten fresh ones.
This is not a caution about post-hoc analysis alone; one of them had **passed a
pre-registered bar**.
- **Example (post-hoc):** §1615's signal-to-noise candidate, chosen from four
  options on the same 10 points after the registered hypothesis failed, gave
  rho = **.7333, p = .0201**. On 10 disjoint classes: **.0788, p = .84**.
- **Example (PRE-REGISTERED, and the more alarming one):** §1614's
  rho(class n, null share) = **.6727 at p = .0192** was registered in advance,
  passed, was scored as "REAL at power", and was mirrored to the theseus registry.
  On disjoint classes: **.0182 at p = .96**. Withdrawn.
- **Near non-example:** the 1/sqrt(n) noise-floor law reproduced at rho .988-1.000
  across BOTH 10-class sets. Mechanistic laws with a known functional form survive;
  free-form rank correlations on 10 points do not. The difference is whether the
  relationship was PREDICTED from structure or SELECTED from data.
- **Rules:** (1) any correlational claim on ~10 units is provisional until it
  replicates on a disjoint set of comparable size and range -- budget the
  confirmatory run when you plan the first one; (2) never mirror such a claim into
  the registry before it replicates (§1614 was mirrored and had to be withdrawn);
  (3) a COMPARATIVE bar needs an absolute floor -- §1616's "SNR beats n" passed
  vacuously because |.0788| > |.0182| while both were indistinguishable from zero.

## 18. Splicing machinery by LINE RANGE truncates functions, and a name-check will not see it (§1617-era)
Reusing a working harness by `sed -n 'A,Bp'` is fast and has now silently cut a
function in half. The undefined-NAME gate cannot catch it, because the function
is still *defined* — just wrong.
- **Example:** `digits_head_dispute.py` spliced lines 17-40 of `channel_budget.py`
  to reuse `rx()`. `rx`'s `return v` is on line 41. The gate reported
  `rx in module: True` and no undefined names, the script passed `ast.parse`, and
  it died on the GPU with `AttributeError: 'NoneType' object has no attribute
  'to'` — because `rx()` silently returned None for every class mask.
- **Near non-example:** splicing a whole contiguous BLOCK located by its own
  markers (`src.index('def foo')` to `src.index('def bar')`) is safe; it is the
  hard-coded numeric range that rots the moment the source file shifts by a line.
- **Rules:** (1) splice by MARKER, never by line number; (2) add a return-check to
  the pre-queue gate — every function that callers use as a value must contain a
  `return <value>`; (3) the gate must test SEMANTICS it can afford, not just
  names. AST-clean + names-defined has now passed three broken scripts in one
  session (undefined `beats`, dead `if False else`, truncated `rx`).

## 19. Registered predictions live in TWO places — update both or the registration is theatre (§1620)
The docstring is the registration; the scoring code is what actually runs. Adapt a
harness and change only one, and the run reports bars nobody registered.
- **Example:** `published_vs_null_fineweb.py` was adapted from
  `writer_floor_absmass.py`. I rewrote the docstring's three predictions for the
  new question and left the scoring block untouched, so the artifact recorded
  `pred_a_question_share_gap_le10_2of3` — a stale bar from the parent script — and
  printed **0-for-3**, while the actual registered bars scored **2-for-3**. Both
  numbers are in the same JSON.
- **Near non-example:** it was recoverable ONLY because every raw share was
  printed (LESSONS 16) and the registered bars were simple arithmetic on published
  values. A registration involving thresholds computed inside the run would have
  been unrecoverable.
- **Rule:** add a gate arm that extracts the `pred_*` key names from the results
  dict and requires each to appear in the docstring's "Registered predictions"
  block. A name present in one and absent from the other is a FAIL.

## 20. Before comparing to a published number, re-derive WHAT IT MEASURES — not just how it was computed (§1620-§1623)
Matching the statistic, the corpus, the rows and the sample size is not enough if
the quantity itself differs.
- **Example:** chasing §1597's .718 I eliminated corpus (§1620), sample size
  (§1621), row skip (§1622) and exact config (§1623), measuring .525-.548 every
  time and coming within one run of reporting a discrepancy in published work.
  The cause was that `slice_writers.py` STOPS its forward at the slice site and
  attributes the SITE'S INPUT over 24 upstream components, while my harness ran
  the full stack and attributed the FINAL residual over 37. Matching that gave
  **.7179 against .718** — exact.
- **Near non-example:** the within-run λ-vs-null comparisons stayed valid
  throughout, because both arms measured the same quantity on the same rows. A
  quantity mismatch poisons comparisons ACROSS setups, not within one.
- **Rules:** (1) before quoting your number against a published one, read the
  source for its projection target, its component set and its forward extent —
  three things, not one; (2) when reusing a harness for a NEW question, re-derive
  the component set and projection point rather than inheriting them (this one
  came from §1601, where they were correct for a different question);
  (3) a stable number reproduced across four configurations is evidence your setup
  is CONSISTENT, not that it is RIGHT.

## 21. A FAILING GATE IS A HYPOTHESIS ABOUT THE SCRIPT, NOT A VERDICT ON IT — check the gate first (§1625, §1627)
Twice in one session my pre-queue gate reported FAIL on code that was correct. Both
times the fault was in the gate's own parsing, and both times the tempting move —
"fix" the flagged script — would have damaged working code and possibly introduced
a real defect while believing I was removing one.
- **False positive 1 (§1625):** the prediction-key arm used
  `'(pred_[a-z0-9_]+)':` and could not match the uppercase in
  `pred_b_smaller_than_S1613_1417`, so it saw 2 keys where 3 existed and failed the
  distinctness check. Fix: `[A-Za-z0-9_]+`.
- **False positive 2 (§1627):** the undefined-name arm collected module-level names
  from `ast.Assign` targets but only handled `ast.Name`, so the tuple assignment
  `CHUNKS, ROWS_PER_CHUNK = 3, 160` registered NO names and both were reported
  undefined in `main()`. Fix: handle `ast.Tuple` targets in every place the gate
  collects bindings — module level, function level, `for`, and comprehensions.
- **Why this is the dangerous direction.** A gate false NEGATIVE costs one bad run.
  A gate false POSITIVE costs correct code, and it arrives wearing the authority of
  a check — the instinct is to trust the tool over the script. LESSONS 18 built the
  gate to stop me shipping broken scripts; it can equally stop me shipping working
  ones.
- **Rules:** (1) on FAIL, reproduce the specific finding by hand before editing the
  script — read the actual line the gate objects to; (2) if the gate's complaint is
  about a NAME or a PATTERN, suspect the gate's parser first, since those arms are
  string-matching and string-matching is where it is thin; (3) fix the gate and
  re-run rather than working around it in the script — a gate weakened to pass one
  script silently stops protecting the next one.

**LESSONS 21 addendum — the gate is now a FILE with a REGRESSION TEST (`ops/gate.py`).**
Two more false positives appeared after LESSONS 21 was written, taking the total to
four: (3) the undefined-name arm did not bind NESTED `def`s or LAMBDA arguments, so
it flagged `hook`, `add` and a `sorted(key=lambda h: ...)` variable; (4) the
"function used as a value must return something" arm used a SUBSTRING match
(`f'{name}(' in s`), which flagged `capture_fwd` — a function called as a bare
STATEMENT. Fixed by walking `ast.Call` nodes and treating a call that is the whole
of an `ast.Expr` as statement-use.
- **The fix that actually matters is the harness, not the four patches.** The gate
  now lives at `ops/gate.py` and is run BOTH directions before it is trusted:
  against every script already known good (must all PASS) and against deliberately
  broken copies (must all FAIL). Defect 4 was caught by that regression pass within
  seconds of writing it, not by inspection.
- **Rule:** a checker with no negative control is an assertion, not a test. Any new
  gate arm ships with a broken-copy that it must catch, and the whole gate is
  re-run over the known-good corpus before use.

## 22. A WRITE THAT MATCHES NOTHING SUCCEEDS SILENTLY — verify the mutation, not the call (§1632)
Twice in one session an operation reported success while doing nothing.
- **Registry flag (§1632):** I looped `for k, v in d.items()` looking for
  `v.get('section') == 'S1613'` to attach a pending-withdrawal flag. The entry is
  nested at `_slice_writer_graph/share_null_calibration`, not at top level, so the
  loop matched zero entries, the script printed "registry updated", and the flag
  warning readers off a refuted claim was never written. Only a separate
  `flag placed: {bool}` assertion caught it.
- **Artifact record (§1632):** `NAMED` became `[]` after a rewiring while the
  results dict still did `{c: ... for c in NAMED}`. The run scored correctly and
  wrote a file, but `fraction` serialised as `{}`. Nothing errored.
- **Why this class is nastier than a crash.** A crash is loud and stops the
  pipeline. A silent no-op leaves a committed artifact or registry that LOOKS
  updated, and the failure is only discovered later by someone trusting it.
- **Rules:** (1) never report a mutation from the fact that the code ran — assert
  the post-condition (`'X' in obj`, `len(matched) > 0`, `bool(d['fraction'])`) and
  print it; (2) any loop that selects by a key or path prints how many items it
  matched; (3) `ops/gate.py` now catches the empty-literal case for results dicts —
  the registry case is not statically checkable, so it needs the assertion habit.

## 23. A CONTROL MUST MATCH THE CLAIM ON CLASS TYPE, NOT JUST ON CELL AND CONFIGURATION (§1633-§1637)
Six axes matched and one missed was enough to invert a conclusion twice in one hour.
- **§1634's lesson:** a control must live at the same CELL. §1630 tested a claim about
  pronouns@mlp17 using mlp11 data and "restored" a reading that direct mlp17
  measurement then refuted.
- **§1637's lesson, one level deeper:** it must also be the same CLASS TYPE. §1633
  compared `question` (sentence-terminal PUNCTUATION) against five FUNCTION WORDS and
  found a margin of 13/60. Against `period` — punctuation, same cell, same rank, same
  TOP, same rows, same seeds — the margin is **2/60**. The depth profile of separation
  turns out to be class-type dependent (function words bottom at mlp11, punctuation
  peaks near it, capitalised tokens rise monotonically), so a function-word control
  was measuring a different curve.
- **Why it was invisible:** the controls WERE rigorous on every axis I thought to
  check — matched rank, matched TOP, identical rows, 20 independent bases, same
  statistic, same site. Rigour on six axes reads as rigour, and the seventh never came
  up until a class of a different type was run for an unrelated reason.
- **Rules:** (1) before running a control, write down what KIND of thing the target is
  — its token class, its frequency band, its syntactic role — and pick controls of the
  same kind; (2) when a margin is the headline, run at least one same-type control
  before reporting it; (3) if the available controls are all of one type, say so in
  the write-up as a limitation, because that is exactly what §1636 did and it is why
  §1637 got run at all.

## 24. ASK WHETHER A TOLERANCE GATES A DETERMINISM CHECK OR A DIFFERENT COMPUTATION — the answer inverts the rule (§1639 aftermath)
PRE-FLIGHT E says never a fixed absolute tolerance on a spectrum. That is right for
comparing two DIFFERENT computations of the same quantity, and wrong for a replay or
determinism check on identical arithmetic.
- **What happened:** Codex's site-0 failed a row-CE gate at `2.06e-5 > 2e-6`. I
  computed the ratio to their baseline CE (44.5 float32 eps observed against a
  4.3-eps bar), concluded an 18-layer float32 forward cannot deliver 4.3 eps, and told
  them the gate was too tight. Their diagnostic then measured same-device CUDA-float32
  drift at **9.64e-7** — 2.1 eps, comfortably inside the bar I had called unachievable.
  The real fault was mixed CPU/CUDA scoring, which is genuinely different arithmetic.
- **The error:** I modelled the check as two independent noisy computations and
  reasoned about accumulation. A same-device replay of the same kernels in the same
  order is near-deterministic, so the tolerance was gating DETERMINISM. Accumulated
  round-off is only the right model once the two sides differ in arithmetic.
- **What worked anyway:** the falsifier. I wrote "if the empirical floor comes back
  near 1e-6 rather than 2e-5, the gate was right" — it came back at 9.64e-7 and the
  question was settled without argument. Register a falsifier whenever you contradict
  someone's gate.
- **Rules:** (1) before judging any tolerance, ask what is on each side — identical
  arithmetic replayed, or two different computations of one quantity; (2) for the
  former, tight absolute bars are legitimate and a failure is real signal, usually a
  device/order/dtype mismatch; (3) for the latter, scale by magnitude and precision as
  PRE-FLIGHT E says; (4) when contradicting a collaborator's gate, state the number
  that would prove you wrong.

## 25. A NEWLINE-SEQUENCED GUARD IS NOT A GUARD — bash survives the failure and runs the next line anyway (§1646)
The pre-queue gate and the durability commit both silently did not happen, and the run
went to the GPU regardless.
- **What happened:** the command was `python3 ops/gate.py X.py` / `cd <BQ>` /
  `echo path >> queue.txt` / `git add ...`, sequenced by newlines. The gate line ran
  from the WRONG cwd (`ops/gate.py` not found), the `cd` came after it, and the later
  `git add` then used a path relative to the new cwd and matched nothing. Bash reported
  both failures and continued. The `echo` in the middle succeeded, so the runner
  executed an **ungated** script whose source was **uncommitted** on a box that is not
  volume-backed. Retroactively the gate passed and nothing was lost — the exposure was
  real regardless.
- **Why it is worse than it looks:** the queue append is the irreversible step. Once the
  path is in `queue.txt` the runner will execute whatever is there, so every guard must
  complete BEFORE it, not merely appear above it.
- **Rules:** (1) `cd` FIRST, as its own statement, before anything path-relative;
  (2) chain guard steps to the irreversible step with `&&`, never newlines —
  `cd <dir> && python3 ops/gate.py X.py && git add X.py && git commit -q -m ... && echo "$PWD/X.py" >> queue.txt`;
  (3) treat a non-zero exit anywhere in a queueing command as "the run did not happen"
  and re-check, rather than reading the science output and moving on — I read the
  RESULT of this run before noticing two lines above it had failed.

## 26. A RESULT THAT IS NOT IN FINDINGS OR THE REGISTRY WILL BE RE-RUN (§1326 duplication, 2026-08-27)
I did consult the record before building — and still duplicated a measurement.
- **What happened:** asked to work bottom-up on mlp0, I searched the ledger, found
  §1324 (mlp0 ceiling 86.3%) and §780 (mean-table rank 22.7), judged the *rank* axis
  unmeasured, and built a run for it. That run recomputed mlp0's mean-ablation stake
  from scratch — badly, with an unweighted token-mean constant and 23.4% of eval
  positions falling back to a ZERO vector — when `opt_ablation_consts_all.pt` had held
  optimal constants for all 198 components since 02:30, and §1326 had held the full
  18-module dossier all along. Logan caught it in one line.
- **Why searching the ledger was not enough.** BILIN18_CONNECTION.md is ~40,000 lines.
  A grep for the right concept returns the sections that MENTION it, not the one that
  SETTLED it, and §1326's headline ("THE FULL-DEPTH MLP LADDER: THREE REGIMES") does
  not contain the words I searched for. Consolidation is what makes a ledger that size
  searchable; without it the ledger is an archive, not a reference.
- **Rules:** (1) before building anything, check FINDINGS and the registry FIRST and
  the ledger second — if the answer is not in the consolidated docs, ALSO ask whether
  it is there under a headline you would not have guessed; (2) grep the ARTIFACT
  DIRECTORY, not only the prose — `ls *consts*.pt`, `ls *scoreboard*.json` would each
  have stopped this in one command; (3) when you find a result buried in the ledger,
  SURFACING it is itself the deliverable — the next agent will otherwise pay the same
  cost; (4) a duplicated measurement that DISAGREES with the original is worse than
  useless, because it invites reconciling two numbers when one of them is simply wrong.

**LESSONS 25 RECURRENCE (same day, 16:30) — and this time it was the VERIFICATION that
went unguarded.** Assembling `_front_band_account` I wrote
`python3 - <<PY ... PY` on one line and `git add ... && git commit ... && git push` on
the next. The python's post-condition assertion FAILED and the commit ran anyway,
because newline-separated statements do not gate each other. The assertion turned out
to be wrong rather than the data (I miscounted expected dict keys, forgetting a
`reading` field), so nothing bad landed — but I committed a registry entry whose
verification had just failed, and only noticed because the traceback and the word
PUSHED appeared in the same output.
- **The sharpened rule:** LESSONS 25 said `&&`-chain the guard to the irreversible
  step. That applies to VERIFY-then-COMMIT exactly as much as to GATE-then-QUEUE. A
  post-condition check that cannot stop the commit is decoration.
- **Correct form:** `python3 verify.py && git add X && git commit -m ...` — one chain,
  no newlines between the check and the thing it guards.

## 27. POSITION-WISE MASKING AT SCORING TIME CANNOT ISOLATE A POSITION-WISE INTERVENTION IN A TRANSFORMER (§1659-§1661)

I measured mlp0's token-table ceiling with a table that covered 76.6% of eval positions.
The uncovered positions got a fallback value. I then excluded those positions from the CE
average, believing that isolated the coverage problem.

It does not. The substituted forward pass still ran with WRONG mlp0 outputs at the
uncovered positions, and attention in layers 1-17 mixed that error into the predictions
at the covered positions. The score-time mask removes the directly-damaged rows and
leaves all of the indirect damage.

- **How wrong it was:** the frozen-attn0 arm has a KNOWN ANSWER of 1.0. Substituting
  everywhere and masking the score gave 55.83%. Substituting only where covered gave
  100.00% (ce_table == ce_live to five decimals). A 44-point error, in a quantity whose
  true value was derivable in advance.
- **Why it fooled me twice:** the first version's headline moved 25 points in the
  DIRECTION OPPOSITE the hypothesis, which reads as a strong negative result rather than
  as a bug. The second version's "obvious fix" moved it 6 points and still read as a
  clean negative. Neither looked like an artifact from the outside.
- **What caught it:** an instrument check with an answer known before the run. Not a
  sanity check on the output's plausibility -- a quantity I could derive analytically
  (attn0 constant + position-wise MLP => mlp0 is a token function => a covered table is
  exact) and refuse to proceed past. It is the only reason the 25-point "finding" is not
  in FINDINGS.
- **The rule:** when substituting position-wise, FIX THE FORWARD PASS, do not repair the
  score. Apply the intervention only where it is valid and leave the module live
  elsewhere (`torch.where(valid, substitute, out)`). In a model with attention, no
  post-hoc row mask can undo a bad value that has already been mixed forward.
- **Scope:** this contaminates every table-ceiling in this project that used an
  unseen-token fallback and substituted at all positions -- they are UNDERSTATED. On
  mlp0 the understatement was 15.9 points (74.42% -> 90.27%).

## 28. A SUBSTITUTION FITTED AGAINST THE REAL MODEL AND APPLIED JOINTLY COMPOUNDS OFF-DISTRIBUTION — and past some number of sites it goes negative (§1668)

Least-squares linear maps for bilin18's MLPs, each fitted from that module's real input to
its real output, then installed at every site at once:

| sites | joint ceiling |
|---|---|
| 12 (middle band) | 62.33% |
| 18 (whole stack) | **-42.99%** |

A negative ceiling means the substituted model is WORSE than one with every MLP replaced
by a constant. Nothing is wrong with the fit: the middle-band arm replicated a previous
run to the digit in the same execution.

- **Mechanism:** site L's map is fitted on the input distribution of the REAL model, then
  applied in a model where sites 0..L-1 are already substituted. Its input is no longer
  the one it was fitted on, and the error at each site enlarges the mismatch for every
  site above it.
- **It is not all-or-nothing, and the warning sign is NON-MONOTONICITY IN RANK.** The
  front band's curve ran 30.1%, -9.0%, 9.8%, 12.3%, 52.6%, 68.7% as rank increased. A
  rank curve that dips below its own rank-1 value is reporting compounding, not
  dimensionality, and no reading should be taken off its interior.
- **What stays usable:** the full-rank arm, which has no truncation error and so isolates
  compounding alone, and any single-band arm small enough to be stable. What is not usable
  is a truncated interior point in a compounding arm, or any cross-band comparison that
  includes a failed one.
- **The rule:** when substituting at many sites jointly, check the rank curve for
  monotonicity and the ceiling for sign BEFORE reading any number off it. If either fails,
  the arm measures the compounding, not the model.
- **Prior art in this ledger:** §546 found the opposite remedy also fails -- refitting a
  block-1 table against a model with block 0 already substituted cost +1.0647 against
  +0.6654 for the naive fit. Neither naive nor refitted composition is safe by default.

## 29. A FULL SET OF PASSING PREDICTIONS IS NOT EVIDENCE THAT THE CODE RAN (§1680-§1681)

A rank sweep returned 60.81% at every rank from 8 to 1152, identical to five decimals, and
all three registered predictions PASSED: the curve was "monotone" (it was constant), rank
128 reached "100.0% of full rank", and the identity arm "reproduced" the reference exactly.

The truncation had never been inserted. The script was built by string-patching a previous
one and the `.replace()` for the truncation matched nothing (LESSONS 22, recurring). My
post-build check asserted the file parsed and that a marker string was present -- it did not
assert the EDIT had landed. The variable it set was written every iteration and read nowhere.

- **Why the usual defences all failed.** Registered predictions did not fire because every
  one of them was a claim about RELATIONSHIPS BETWEEN ARMS, and a no-op makes those
  trivially true -- monotone, ratios of 100%, identity reproduced. The known-answer
  instrument check did not fire either: the identity arm is precisely the arm a no-op gets
  right. Neither mechanism is designed to detect that nothing happened.
- **What actually caught it:** noticing that five ranks cannot return identical values. That
  is a person looking at output, not a control, and it would not have fired on a subtler
  no-op -- a truncation applied at one site only, or to the wrong matrix, produces a
  plausible curve.
- **Rule 1, at build time:** when patching one script into another, assert the count of
  every anchor BEFORE replacing (`assert s.count(old) == 1`). Asserting a marker is present
  afterwards proves the marker, not the edit.
- **Rule 2, at run time:** any sweep over a parameter must assert its arms are not all
  identical. One line, catches the whole class.
- **The general shape:** predictions constrain the SCIENCE, instrument checks constrain the
  MEASUREMENT, and neither constrains whether the INTERVENTION was applied. That needs its
  own check, and the cheapest one is that varying a parameter must vary the output.

**LESSONS 29 addendum — the BASELINE needs a known-answer check too (§1694-§1695).** Every run
in this arc since §1659 carries an identity arm with a pre-derivable answer. All of them check
the SUBSTITUTION: that a full-rank map recovers its projection, that a frozen-attention table
is exact, that k=all keeps everything. None of them constrains the BASELINE.

A ceiling is `(ce_const - ce_program) / (ce_const - ce_live)` -- three measured numbers. I
registered constant-ablation hooks directly on the modules and measured `ce_live` THROUGH them,
getting 8.86042 where the arc's own value is 3.29205. Every stake went negative, every ceiling
NaN. The identity arms would have passed had the run reached them, because the substitution
was fine.

- **The rule:** a known-answer check on one term of a ratio is not a check on the other two.
  Pin `ce_live` against its published value the same way the identity arm pins the substitution.
- **Why it is easy to miss:** the baseline feels like setup rather than measurement, so it does
  not attract a control. It is a measurement.
- **What actually caught it:** 8.86 being visibly wrong to a reader of the ledger. That is the
  same luck that caught §1681's no-op, and luck is not a control.

## 30. A QUEUED EXPERIMENT CAN VANISH SILENTLY — verify the append, and verify it LATER (§1699 aftermath)

`whole_model_heldout.py` was gated, committed, pushed, and appended to `queue.txt` with the
append reporting success. It then never ran: no log in `runlogs/`, no line in
`runlogs/_completed.txt`, and no entry left in `queue.txt`. It was silently dropped, almost
certainly a race between my `echo >> queue.txt` and the runner popping the first line and
rewriting the file.

- **Why it is dangerous:** this is the exact failure the loop's "never let a lane idle" rule
  exists to prevent, and it is invisible. A dropped entry looks identical to a queued one that
  has not started yet, so a wait-for-the-log loop just hangs (mine did, for a full 600s
  background timeout) and the natural reading is "still running".
- **The tell:** the entry is absent from `queue.txt` AND has no log AND has no
  `_completed.txt` line. Any two of those three are normal at some point in a run's life; all
  three together mean it was lost.
- **Rule 1:** after appending, re-read `queue.txt` and assert the path is in it (or that a log
  already exists). One line, catches the race at append time.
- **Rule 2, the one that actually mattered here:** when a run seems slow, check for the
  three-way absence before waiting again. I waited 600s on a log for a job that had already
  been dropped.
- **Related:** LESSONS 25 (a newline-sequenced guard is not a guard) and LESSONS 29 (a passing
  prediction does not mean the code ran) are the same shape -- an operation reporting success
  is not evidence the intended effect happened.

**LESSONS 30 RECURRENCE (same evening, second occurrence) — the mechanism, and why it is a
COORDINATION problem not a bug to fix alone.** `whole_model_v1_floor` and `whole_model_heldout`
were both dropped a second time: absent from `queue.txt`, no logs, no `_completed.txt` lines —
the three-way absence LESSONS 30 names. The cause is now clear and is not a race with the runner
alone: `queue.txt` is a SHARED file with two writers. The runner pops the first line and rewrites
the file, and both agents append to it. Any read-modify-write by either party between another's
append and the rewrite silently discards the other's entries.

- **Why it must not be "fixed" unilaterally:** the obvious repairs (a lock, per-agent queue
  files, an append-only journal) all change an interface the other agent is actively using. A
  one-sided change would break their lane, which is a worse failure than the one being fixed.
  Raised on the board instead.
- **What IS safe to do alone:** re-append and verify immediately (rule 1), and check the
  three-way absence before waiting again (rule 2). Both applied.
- **Cost so far:** two experiments dropped, one 600-second wait on a job that did not exist.
  Nothing scientific was lost because nothing had run, but the lane sat idle twice, which is the
  specific failure the loop exists to prevent.

## 31. A PREDICTION THAT CANNOT FAIL IS NOT A PREDICTION — three mechanisms, one consequence (§1681, §1684, §1705)

Three registered predictions passed this arc without testing anything. Different causes, same
failure, and none was caught by the usual defences:

1. **§1681 — the code did not run.** A string-patch matched nothing, so a rank sweep returned an
   identical ceiling at every rank. All three predictions were claims about RELATIONSHIPS BETWEEN
   ARMS — monotone, 100% of full rank, identity reproduced — and a no-op makes every such
   relationship trivially true.
2. **§1684 — the arms were nested by construction.** With every attention write pinned to a
   constant, `v1` has no causal route to the logits, so "ablate both" was identically "ablate the
   write". The prediction "the two paths are not additive by >= 10%" was satisfied by a
   structural identity. Each arm individually computed exactly what it claimed to, so no
   instrument check fired.
3. **§1705 — the bar was one-sided and the sign was the question.** I wrote
   `excess_fraction < 0.340` expecting a smaller POSITIVE excess. The observed −0.201 cleared it
   by being NEGATIVE — the prediction assumed the very sign it was meant to test.

- **The common structure:** in each case the prediction was satisfiable WITHOUT the phenomenon
  being present. That is a property of how the bar was written, not of the science, and it is
  invisible from the pass/fail line.
- **Why the standing defences miss it.** Registered predictions constrain the science and
  known-answer identity arms constrain the measurement, but neither asks whether the bar is
  reachable from the other side. A no-op gets the identity arm RIGHT. A nested arm computes
  honestly. A one-sided bar is arithmetically correct.
- **The check, and it costs one sentence per prediction:** before registering, name the specific
  observable result that would make this bar FALSE, and confirm that result is achievable under
  the run's own design. Applied to the three: §1681's "monotone" is unfalsifiable when every arm
  can return the same number, so the guard is "assert the arms are not all identical"; §1684's
  joint arm cannot differ from the write arm once the write is constant, so the guard is to check
  the arms are causally distinct before scoring their difference; §1705's bar admits both signs,
  so the guard is to write it two-sided or to bound the magnitude.
- **What it is NOT:** a reason to weaken bars. All three of these were strict. Strictness does not
  help when the quantity cannot move.

## LESSON 32 — one control is not a control

A specificity ratio with a single denominator is a coin flip wearing a ratio's clothes. §1724
scored every registry circuit against one deterministically-picked matched-size control set and
concluded the median circuit "matters less than arbitrary". §1725 replaced the single pick with
twelve random draws and two headline rows **reversed**: mlp0 from 0.12 to 14.81, mlp1 from 8.25 to
151.56. The 0.12 was not a fact about mlp0; it was a fact about mlp1 having been drawn as its
control.

The tell was visible in the v2 output before any second run and I read past it: **four of fifteen
rows were exact reciprocal pairs.** 4.3301/3.5570 = 1.2173 next to 3.5570/4.3301 = 0.8215;
7.0213/0.8514 = 8.2468 next to 0.8514/7.0213 = 0.1213. Reciprocals in a table of ratios mean the
numerator and denominator are drawn from a two-element pool — the control set is not a sample of
"elsewhere", it is the one other thing.

Two rules from this:
- **A denominator drawn once is a denominator with no error bar.** If a comparison set can be
  sampled, sample it more than once and report the percentile, not just the ratio. If it *cannot*
  be sampled more than once — a set naming all 18 sites of one kind has exactly one matched-size
  elsewhere — report that the percentile does not exist rather than a degenerate 0 or 1.
- **Scan a results table for algebraic coincidences before interpreting it.** Exact reciprocals,
  exact complements, and rows that sum suspiciously to a round number are usually the instrument
  showing through, not the model.

Related: LESSONS 29 (a full set of passing predictions is not evidence the code ran) and LESSONS 31
(a prediction that cannot fail is not a prediction). This is the third member of the family: **a
comparison whose baseline has one degree of freedom cannot fail informatively either.**

## LESSON 33 — an aggregate over registry entries is an aggregate over prose

§1725 reported that 5 of 7 direction-annotated circuits were claim-consistent, clearing a 2/3 bar.
Three of those seven entries name the **same four components** and were one measurement counted
three times; deduplicated the tally is 3/5 and the predicate fails (§1726, found by Codex).

The registry is a list of *claims*. A harness that scores *components* produces one number per
distinct component set, however many entries point at it. Averaging over entries weights each
measurement by how many times someone wrote it up.

- **Before any aggregate, collapse rows to distinct measurements and print the collapse ratio.**
  If sixteen rows become nine, the denominator is nine and the reader should be able to see that.
- **Write the bar over the deduplicated reading.** §1725's bar said "circuits", which is literally
  registry rows, so it passed as written — and passing as written on a meaningless denominator is
  worse than failing, because nothing flags it.
- **Naming a bias does not immunise the next number against it.** §1722 recorded duplicate
  component sets as a known limitation of this exact harness. The inflated predicate came four
  sections later, in the same harness, and I did not check.

Sibling of LESSONS 32 (one control is not a control): both are cases where the *denominator* was
the unexamined part while all the attention went to the numerator.

## LESSON 34 — a partition needs a known-answer check, not a sum check

§1727–§1729 measured constant-ablation damage decomposed into three target-token classes and
certified two registry entries from it. The `induction` mask compared `j < p` where the docstring
said `p < j`, so it searched **future** positions; all three cells were contaminated and all three
sections are void (§1733).

Four controls passed. None of them could have failed:

- **baseline CE** and **total removal** are pooled over classes — identical under *any* partition.
- **"class counts sum exactly to the scored count"** verifies **exhaustiveness**, not correctness.
  Three arbitrary buckets pass it. This is the one that reads like a class check and is not.
- **"the joint ratios reproduce the previous run's"** reproduces your own wrong computation.
  LESSONS 29 again: an arm that confirms a computation against itself confirms nothing.

The rules:

- **Every derived population needs one hand-built example with a known answer.** Four tokens is
  enough: `[5,7,5,7]` predicting `[7,5,7,9]` has induction at position 2 and must not have it at
  position 0. Three lines, and it fails loudly under an inverted mask.
- **A sum, a total, or a pooled reproduction cannot validate a split.** If the quantity you check is
  invariant to the partition, it is not a check on the partition.
- **Write the axis convention next to the comparison, not in the docstring above it.**
  `causal[j, p] = (p < j)` as a comment on the line, so the reader compares one symbol to one symbol.
- **Do not copy a derived-population function into a third script.** It was carried forward by
  copy-paste into three files; one shared module with one test would have caught it once.

Sibling of LESSONS 32 (one control is not a control) and 33 (an aggregate over registry entries is
an aggregate over prose). All three are the same shape: the numerator got the scrutiny and the
**population, denominator, or comparison set** was assumed.

## LESSON 35 — a guard on a denominator manufactures a number where there was none

§1740 printed `PROG/OAT = 308,477,470.98x`. The denominator was **−0.0008**, and the code said
`g['PROG'] / max(g['OAT'], 1e-9)`. The guard existed to avoid a division-by-zero crash and instead
converted "this ratio is undefined" into a large, confident, printable number that then went into a
registered prediction (`ratio non-increasing in K`) and made it unscoreable.

This is the third time in this arc that a denominator did the damage while the numerator got the
attention (LESSONS 32: one control is not a control; LESSONS 33: an aggregate over registry entries
is an aggregate over prose), and the second time specifically that a **zero-crossing denominator**
did it — §1728's per-site ratios blew up to −3.6 and −5.2 at three attention sites, and §1735
replaced them with a difference for exactly this reason. I wrote a fresh one four sections later.

- **Before dividing, ask whether the denominator can be zero or negative.** If it can, the ratio is
  not a statistic on that domain. Report the **difference**, which is defined everywhere and, for
  quantities in nats, is the unit anyone actually cares about.
- **Never floor a denominator to make a division succeed.** A crash is information; `max(x, 1e-9)` is
  a fabricated result. If a guard is genuinely needed, emit `None` and print `n/a`, so the row is
  visibly missing rather than invisibly wrong.
- **A guard added for robustness is a silent change to what the number means.** It should be as
  suspect as any other transformation of the data, and it belongs in the same review as the metric.

### LESSON 31, addendum (2026-08-28) — I wrote a nested arm four sections after naming the defect

LESSON 31 lists three ways a prediction fails to be a prediction: a no-op, **nested arms**, and a
one-sided bar. §1743's `pred_b` asked whether local search's improvement transfers to a second role.
`pred_a` found no improvement existed. **pred_b was then decided by pred_a's outcome and could not
have passed**, so it tested nothing and was scored as a failure anyway.

Two rules from it:

- **Before registering an arm, ask what happens to it under every outcome of the other arms.** If
  some outcome of arm A makes arm B unanswerable, B must be declared conditional and reported `n/a`,
  not scored.
- **An under-budgeted arm is the same defect in a different place.** §1743's `pred_c` asked whether
  two starts converge to the same set, with `MAX_SWEEPS = 2`. The random start was still improving by
  +0.21 nats when the cap hit, so the arm could not answer its question either. Setting a compute cap
  is setting the arm's power; do it deliberately or the arm is decoration.

Naming a defect in LESSONS does not stop you committing it. What stops it is checking the arms
against each other at registration time, which takes about a minute and is now part of writing the
prediction block.

## LESSON 36 — never compare a rounded number against an unrounded one

§1744's `pred_b` asked whether three converged local optima were strictly worse than a reference. It
printed **True**. They were not worse — they were the **same six sites**, byte-for-byte the same
allocation. The comparison was `round(curv, 5) < gv - 1e-9`: a value rounded to five places against a
reference carrying full precision. Both are 1.20366; the rounded one sits about 5e-6 below the other,
and a strict inequality with a 1e-9 tolerance sailed through.

- **Round for display, never for comparison.** Keep full precision in the variable the predicate
  reads, and round only inside the f-string. Storing `round(x, 5)` into the results dict and then
  scoring off that dict is how the two got mixed.
- **When two things may be identical by construction, compare the THINGS, not their scores.** The
  right predicate here was `set(final) != set(greedy)`, which is exact, cheap, and could not have
  been fooled. A float comparison was answering "are these the same allocation?" the hard way.
- **A tolerance smaller than your rounding step is decorative.** `1e-9` next to `round(x, 5)` looks
  careful and is meaningless: the rounding introduces error four orders of magnitude larger than the
  tolerance is guarding against.

Sibling of LESSON 35 (a guard on a denominator manufactures a number): both are cases where a routine
numerical convenience — a floor, a rounding — silently changed what a registered predicate tested.

### LESSON 31, addendum 2 (2026-08-28) — a second nested arm, six sections after the first

§1749's `pred_b` compared the pass-3 gain against the pass-2 gain. `pred_a` found the pass-2 gain was
exactly zero, so pred_b had nothing to compare and was decided by pred_a's outcome — the same defect
as §1743's, recorded in this file's first addendum, committed again six sections later.

The rule was already written: **check every arm against every outcome of the others before
registering.** Writing it down did not make it happen. What was missing is a step in the routine, so
it is now stated as one: when the prediction block is drafted, walk each arm and ask "if arm A comes
out the other way, does this arm still measure anything?" — and mark any arm that does not as
conditional, to be reported `n/a`.

Also worth keeping from §1749: **an exactly-zero result can be a proof rather than a null.** Three
coordinate-descent passes changed the program by zero to five decimals because a transformer is
causal in depth, so a site's fit depends only on what is compiled below it and one bottom-up pass is
already a fixed point. The identical digits were the evidence; a "no significant change" reading
would have thrown that away.

## LESSON 37 — one observation is not a law, especially when it becomes a design decision

§1751 took a shortcut — one interleaved compile served three correction ranks, with the prefix built
at the top rank — and observed that the rank-128 prefix made the rank-8 program *better*. §1757
reused the shortcut and wrote that observation into its header as an established asymmetry. **It
reversed**: with compressed tables the same shortcut made the rank-8 program worse by 0.325 nats, and
four of six cells in the grid became uninterpretable.

- **An observation from one configuration is a fact about that configuration.** Before it justifies a
  design choice elsewhere, it needs to have been measured in the new configuration, or the choice
  needs to not depend on it.
- **Watch for the sentence "X measured this as helping, so the shortcut is safe."** That sentence
  converts a single data point into a licence. The honest version names the sample size: "§1751 saw
  it help once; direction unknown in general."
- **A shortcut that couples arms is worth its full cost to remove.** Six compiles instead of two
  would have cost eight extra minutes and produced six interpretable cells instead of two.

The control is what caught it — the two cells that were supposed to reproduce a published number came
out 0.33 and 0.76 nats away. Sibling of LESSON 34: a control that can actually fail is the only kind
worth writing.

## LESSON 38 — a correction lands when the last artifact is updated, not when the ledger is written

§1718, §1720 and §1723 corrected one quantity three times. All three corrections went into the
ledger. **None reached the registry**, where three certified entries kept quoting the superseded
figures — including `_CURRENT_HEADLINE_FIGURES`, whose entire purpose is to stop people quoting
superseded figures (§1759).

- **When a number changes, grep for it before closing the correction.** Not for the section that
  introduced it: for the *value*. It takes seconds and it is the only step that finds the copies.
- **Rank artifacts by who reads them.** A ledger is read by its author; a registry is read by
  collaborators and by future runs as ground truth. The one with the wider audience should be
  corrected FIRST, not last.
- **Record superseded values, do not delete them.** A reader who saw the old figure needs to find out
  it moved, which a silent overwrite denies them.

Sibling of LESSON 33: both are cases where the number was right somewhere and wrong where it counted.

## LESSON 39 — write the predicate, then read it back against its own sentence

§1768's `pred_a` said: "the model's own per-token CE is WORSE than the best program's 6.57289. If
FALSE the compiled program beats the model at its own per-token game." The code was
`ceiling > program`. In CE, lower is better — so that expression is False exactly when the ceiling is
**better**, which is the outcome the sentence calls a pass. **The predicate returns False in the case
its own prose describes as success.**

The check was not wrong, the direction was. And nothing caught it: the gate parses prediction keys,
the two-sidedness rule (LESSON 31) is about whether an arm *can* fail, and neither looks at whether
the comparison points the way the sentence says.

- **After writing a predicate, substitute the two outcomes back into the sentence.** "If the ceiling
  is 5.98 and the program 6.57, does my expression return what the sentence calls a pass?" It takes
  ten seconds and it is the only step that catches a flipped inequality.
- **In a lower-is-better metric, name the direction in the variable.** `gap_program_to_ceiling` reads
  unambiguously; `ceiling > program` does not, and it was sitting two lines from a correctly-written
  `gap_program_to_ceiling` in the same run.
- **Report the coded result, not the intended one.** The arm as written failed. Saying "it passed if
  you read it the way I meant" is how a bar stops being a bar.

Sibling of LESSON 36 (rounded vs unrounded) and LESSON 35 (a floored denominator): all three are the
comparison itself being wrong while the measurement is fine.

## LESSON 40 — a strict inequality with no margin is not a bar

§1772's `pred_a` said the tightened bound must score "below 5.97902". It scored **5.97900** — the same
computation by a different code path, 2e-5 away — and the predicate returned True. My own controls in
the same run use a **0.001** tolerance, fifty times larger. So the arm passed on a difference that the
run itself treats as reproduction noise.

- **When a bar is "beat X", state the margin.** "Below X by at least m", with m at least as large as
  the tolerance the controls use for the same quantity. Otherwise the arm is decided by float
  accumulation order.
- **The margin belongs in the predicate, not in the write-up.** Noticing afterwards that a pass was
  noise is better than not noticing, but it means the bar did no work — and the temptation to keep the
  pass is exactly what a bar exists to remove.
- **Reuse the control's tolerance as the floor.** If a run reproduces a published number to within
  0.001 and calls that a match, then 0.001 is the resolution of that quantity in that run, and no arm
  should be decided below it.

Sibling of LESSON 36 (rounded vs unrounded) and LESSON 39 (the predicate pointing the wrong way): all
three are the comparison being wrong while the measurement is fine.

## LESSON 41 — a queue that drops malformed lines idles the lane with no failure to notice

I appended my next experiment to `runlogs/queue.txt` with a `[12:04] ` timestamp prefix. The real
queue is `BQ/queue.txt`, and its contract (`ops/bqrunner.sh`) is: **bare absolute paths, one per
line; a line that is not an existing file is popped and dropped.** So the entry was invisible twice
over — wrong file, and malformed even if it had been the right one.

Caught in one minute at the next tick's ORIENT (`cat queue.txt` empty, no per-script log, runner's
last line still the previous run), so the GPU lost nothing. That is luck, not process: **nothing in
this system reports an unfed lane.** A completed run notifies. A crashed run notifies. A run that was
never queued produces silence that is indistinguishable from a healthy idle, and the runner's canary
keeps the box warm so even the GPU trace looks normal. This is LESSON D — watchers fail in two
directions — applied to the queue: I had been checking "did my run finish", never "did my run start".

**How to apply.** Queueing is not done when the `echo` returns. It is done when
`test -f "$(head -n1 queue.txt)"` passes, and confirmed when `runlogs/<basename>.log` exists or
`runner.log` names the script. Two seconds, and it converts a silent lane into a loud one. Never
decorate a queue line — the contract is the bare path.

### LESSON 41 addendum — my push confirmation could not report a failed push

Same tick, same shape, worse consequence. My push helper is

```
timeout 200 git push -q origin main 2>&1 | tail -1 && { echo TL_OK; break; }
```

A pipeline's exit status is the **last** command's, so this tests `tail`, which always succeeds.
It printed `TL_OK` on a push that had just printed `failed to push some refs`. Every "pushed both
repos" in my summaries has rested on a signal that is structurally incapable of saying no.

Measured immediately, because LESSON B: both repos are now 0 commits ahead of origin, and the earlier
§1788 commits were already on `origin/main` at this tick's ORIENT — so nothing was actually lost. The
helper was wrong for longer than the loss window happened to be.

**How to apply.** Never terminate a check with a pipe. The confirmation for a push is not the push
command's own output but a **separate, positive re-read of the remote**:
`git fetch -q origin && [ "$(git log --oneline origin/main..HEAD | wc -l)" = 0 ]`. Same rule as the
queue above and the same rule as LESSON D: a success signal that cannot be made to print failure is
not a check, it is decoration. When I write one, I owe it one deliberate run against the broken state.

### LESSON 41, third and fourth instances — the same defect in two more directions, same tick

**Third.** Waiting for a rerun I wrote `until grep -q "…exit=" <(tail -5 runner.log)`. The PREVIOUS
run's failure line was still inside that window, so the condition was true before the rerun produced
anything and the wait returned instantly on stale state. The two failures above could not report
failure; this one could not report "not yet". LESSON D says watchers fail in two directions — I hit
both directions inside one tick. Make the predicate name a state that CANNOT already hold: `tail -1`,
a new artifact's mtime, or a counter that must increase.

**Fourth.** This addendum was itself appended to the WRONG FILE. `ops/push_both.sh` runs with
`git -C`, so I stopped prefixing `cd`, cwd drifted to `/workspace/tensor_language`, and
`cat >> LESSONS.md` silently created a stray at the repo root — the same failure as the stray
`ops/BILIN18_CONNECTION.md` earlier in this thread. `>>` creates. It never warns. Caught only because
a later `grep -c` on the real file returned a count that could not be right.

**How to apply, covering all four.** Every one of these was a check or a write that could not
distinguish the state I wanted from a state I already had. For appends to a ledger, use an ABSOLUTE
path — never a relative one, and never trust cwd across tool calls. For checks, ask what the command
prints when the thing has NOT happened; if the answer is "the same thing", it is not a check.

## LESSON 42 — internal consistency cannot detect a changed definition; only cross-run reproduction can

`ops/rank_crossover` (§1792) passed every internal control it had: top-k monotone in k for every arm
and bucket, buckets partitioning every scored position, the fit-row bigram's covered CE reproducing
§1767's 7.88804 / 7.90729 exactly, coverage 5419. It was still measuring the wrong object. Its "LOO
bigram" ranked by `counts + alpha*V*back` at alpha=0.01, where `alpha*V = 503.04` multiplies a
distribution summing to 1 while the actual counts are 1-3 — so the arm was a unigram at the top of its
ranking, and its top-1 came out 10.31% against the 15.97% the same nominal object scored in §1790.

Every internal check was **invariant to the defect**: monotonicity, partition and coverage hold for
any scoring whatsoever, and the CE control tested a *different* table. The only bar that failed was
the one naming a figure produced by an EARLIER RUN OF A DIFFERENT SCRIPT.

**How to apply.** When a run reuses an object from a previous section, one control must be "this arm
reproduces the number that section published", and it must be checked against the published figure —
not against a value recomputed inside the new script, which would move with the defect. A run that
only checks itself can be perfectly consistent and perfectly wrong. Related: [[lesson-34]] (a
partition needs a known-answer check, not a sum check) is the same defect one level down — there the
controls could not fail; here they could fail but not for this reason.

## LESSON 43 — a leave-one-out that decrements the count but not the decision is still a leak

§1790's leave-one-out bigram removed the scored observation from the *count* and then chose with
`torch.where(c1 >= v1, k0, k1)`. Whenever the target was the arm's own top-1 and its decremented count
merely TIED the runner-up, `>=` kept the target — so the observation that had just been withdrawn
still decided the prediction. **30.4 / 30.6 / 41.1% of that arm's correct predictions were held by
such a tie** (§1794), inflating it by 3.52 / 3.75 / 5.75 pp and reversing the published comparison.

The defect is invisible to every check the original run had, because the arm behaved correctly on
every position where the counts were not tied — and with ~6.8 observations per covered type, ties are
the common case, not the edge case.

**How to apply.** A leave-one-out is not done when the count is decremented. It is done when **no
downstream comparison can still resolve in the held-out item's favour**. Concretely: after removing
the observation, take the decision over the whole row rather than patching the top-1 slot, and break
remaining ties by something that cannot see the answer (here, the unigram). More generally, **every
`>=` in a tie-break is an unstated policy**, and when one side of that comparison is the target, the
policy is a leak. Related: [[lesson-42]] — this was found only because a figure from a different
script disagreed; the original's internal controls all passed.

## LESSON 44 — a binary predicate cannot separate three outcomes; print the discriminating curve

§1797's pred_b asked whether a threshold-selection procedure "chose to defer at all", and I registered
its failure as meaning "pred_a failed for want of any candidate rather than because the signal is
uninformative". Both readings map to the same `False`. What actually happened was neither: the grid
held seven deferring candidates, all were evaluated, and **all seven lost to the null**. The signal was
uninformative *and* candidates existed — the stronger conclusion, and the one my registered sentence
explicitly set aside.

I was saved only because the run printed the whole threshold curve rather than the winner. That was a
design habit, not foresight; had the script printed only the selected threshold and its margin, I would
have written up the weaker conclusion and been wrong in the ledger.

**How to apply.** Before registering a predicate, enumerate the distinct states its FALSE could
describe. If there is more than one and they lead to different write-ups, the predicate is not
sufficient: emit the quantity that separates them — the full sweep, the candidate set, the counts —
alongside the boolean. A bar tells you whether you passed; only the curve tells you why you failed.
Related: [[lesson-39]] (write the predicate, then read it back against its own sentence) — this is the
same discipline applied to the FAILURE branch, which I had been checking far less carefully than the
pass branch.

## LESSON 45 — a sweep that does not move the quantity it sweeps is not a test

§1799 varied a ridge by 100x to test whether an accuracy collapse was a conditioning artifact. The
answer came back "no", and I nearly wrote that down. The run's own diagnostic said otherwise: at
n >= 1355 the two ridges produced **identical condition numbers to three significant figures** (1.41e+04,
3.22e+03, 7.12e+02) and identical accuracy at five of six points. The ridge enters as
`ridge * I * (n/D)` against a data term whose eigenvalues are orders of magnitude larger, so "100x the
settled value" was still numerically zero. Where the ridge DID bite -- below n = D, where the matrix is
rank-deficient and the ridge alone sets the smallest eigenvalue -- the condition number fell by exactly
100x and the accuracy did not move at all, which is a second, independent sign that the sweep was not
probing what I thought.

PRE-FLIGHT E already says this: never a fixed absolute tolerance on a spectrum, scale by max|eig|. I
applied it to eigenvalue comparisons and not to a regularisation parameter, which is the same object
wearing a different name.

**How to apply.** Before reading a negative from a sweep, check that the swept parameter actually
changed the intermediate quantity it acts through -- and emit that quantity, not just the outcome
(LESSON 44 again; recording cond(A) per cell is the only reason this was caught). Express any
regularisation as a FRACTION of the relevant spectral scale, never as an absolute number, and prefer a
sweep whose endpoints are known to bracket the behaviour (here: a ridge small enough to be inert and one
large enough to dominate). Related: [[lesson-44]], [[lesson-35]].

## LESSON 46 — scope a control to the cells its prediction actually reads

§1801's pred_d required `cond(A)` to span >=1000x across the ridge sweep **at every n**, a gate written
after LESSON 45 to stop me reading a negative from a sweep that never moved its knob. It failed — not
because the knob failed to turn where it mattered (it turned 3.7e+06x at n = D, the only cell pred_a
tests) but because at n = 1620 the unregularised matrix was *already* well conditioned (cond 3.22e+03)
and there was nothing for the ridge to remove. The gate demanded a large change in a quantity that had
no room to change, in a cell no prediction consulted.

A control that ranges over more cells than its prediction does will eventually fire on one of them, and
when it does it blocks a conclusion the evidence supports. That is the mirror of the failure it was
written to prevent: LESSON 45 was a check too weak to fail, this is a check too broad to pass.

**How to apply.** Write the control over exactly the cells the prediction reads, and no more. If the
prediction is "arm X beats arm Y at cell C", the licensing condition is about cell C — quantify it
there, and report the other cells as diagnostics rather than as gates. When a control does fire, ask
first whether it fired inside the claim's support before treating the claim as unreadable. Related:
[[lesson-45]], [[lesson-40]] (a strict inequality with no margin is not a bar -- pred_b in the same run
failed on 0.04pp wobbles against a 9.45pp effect for exactly that reason).

### LESSON 46 addendum — a knob check must distinguish "did not turn" from "had nowhere to turn"

The gate LESSON 46 was written about fired twice more, in consecutive runs, and gave the same FALSE for
opposite situations. In §1801 `cond(A)` spanned 3.7e+06x at the cell the prediction tested and only
2.9e+02x at a cell no prediction read, so the gate blocked a conclusion the evidence overwhelmingly
supported. In §1803 it spanned 30x everywhere — because at full coverage the matrix is already well
conditioned (cond 324) and there was nothing for a ridge to fix, which was itself the run's headline.

A check that reports FALSE both when a sweep is broken and when a sweep is unnecessary is not
distinguishing the two states it exists to separate (this is [[lesson-44]] again, one level up: the
predicate is binary, the situation has three states).

**How to apply.** Condition the knob check on there being something to fix: require the span only in
cells where the inert-end value itself indicates a problem, and otherwise report the span as a
diagnostic rather than a gate. Emit the inert-end value alongside the span so the two states can be
told apart by eye even when the boolean cannot tell them apart.

### LESSON 41, fifth instance — an unconditional label on a conditional fact

Checking the GPU before queueing, my shell habit was
`nvidia-smi --query-compute-apps=... ; echo "(empty=free)"`. The `echo` prints "free" whether or not the
line above it listed processes. It printed "(empty=free)" while Codex had a collector resident, and I
read the label instead of the output, so a run was queued onto a busy GPU. It fitted (32.6 GiB card,
~22.7 GiB mine plus ~6 GiB theirs) and nothing was lost, but that was headroom, not judgement.

Fixed properly rather than by resolving to read more carefully: `ops/gpu_free.sh` prints actual
occupancy, names every resident process with its command line, and **exits nonzero when the GPU is
occupied**, so `ops/gpu_free.sh && echo <path> >> queue.txt` cannot append onto a busy card. Tested in
both directions per LESSON D: against the live busy state (prints BUSY, names both PIDs, exit 1) and
against a stubbed empty process list (prints FREE, exit 0).

**How to apply.** Never pair a command with a hard-coded description of what its output means. Either
the check exits with a status you can chain on, or it is decoration. Same family as the queue append
and the push confirmation in [[lesson-41]].

## LESSON 47 — points sampled to bracket a feature do not estimate a curve

§1804 probed a cliff at L5/L6 and sampled depths 4, 5, 6 and 13 for that purpose. Its prefix arms were
a by-product, and I noted at the time that they were the most informative thing in the section. In the
next run I registered a prediction about the SHAPE of that curve — that it accelerates with depth —
reading the shape off those four points. The full eighteen-point curve says the opposite: it
decelerates, and the per-layer maximum is at L7-L8, inside the 7-12 range the four points skipped
entirely.

The four depths were well chosen for their own question (two cliff layers, the adjacent control, a
weaker instance). They were a terrible sample for a shape, because they were selected by where a
feature was, not by where a curve needed resolving.

**How to apply.** When a by-product of one experiment suggests a trend, do not register a prediction
about that trend until it has been sampled for that purpose. Ask what the sampling was optimised for:
points chosen to bracket a discontinuity cluster around it and leave the rest of the domain unresolved,
which is exactly the wrong design for estimating slope or curvature. Run the curve, then predict about
the curve. Related: [[lesson-37]] (one observation is not a law, especially when it becomes a design
decision) — this is the same error with four observations and a shape instead of one and a rule.

## LESSON 48 — a marginal along one ordering is not an attribution to a component

§1805 measured a top-down compile curve and reported per-layer marginals from it: "L7 and L8 are the
two most expensive layers to compile", +10.5 and +12.0 points of gap. §1806 measured the mirror curve
from the same build and found its maximum at **L0, +62.6** — a layer whose top-down marginal was
**+0.0**. Same layers, same program, same eval; the two orderings disagree by two orders of magnitude
about which layer matters.

Neither curve is wrong. A marginal is the increment along a particular path through configuration
space, and when components interact strongly the path is most of the answer. §1806's mechanism makes it
concrete: a compiled layer below a live one poisons it, so a layer's measured "cost" depends entirely on
what is compiled around it.

**How to apply.** A difference measured by adding components in one order describes THAT ORDER. Before
writing "component X costs Y", measure it by at least two orderings and require them to agree — that is
what §1806's pred_b was, and it failed, which is the only reason §1805's claim was caught one section
after it was published rather than much later. When they disagree, the honest object is the curve over
whole prefixes or suffixes, not an attribution. Related: [[lesson-33]] (an aggregate over registry
entries is an aggregate over prose) and the §1736-§1739 finding that one-at-a-time ablation overstates
an MLP site by 2.4x and understates an attention site by 2.5x — the same defect, already recorded once
for ablation and now for composition.

### LESSON 41, sixth instance — editing a script after queueing it races the runner

I queued `partial_compile_frontier.py`, then patched it twice more. The runner popped it at 15:35:20;
my final fix landed at 15:35:23. Python snapshots the source at process start, so the run was executing
a version I had already superseded and was guaranteed to die at its last line -- after six minutes of
correct GPU work. I killed it rather than let it finish, but the waste was already committed the moment
I edited a file that was sitting in the queue.

The same tick also ran a version whose `pred_a` could not fail (the settled arm was the minimum-cost
arm in the sweep, so "is it Pareto-dominated" was unfalsifiable) and a version that OOMed holding two
8.3 GiB row banks. Three defects, three runs, one experiment.

**How to apply.** The order is: write, gate, THEN queue -- and once a path is in `queue.txt` or running,
that file is frozen. If a fix is needed, either edit and requeue as a deliberate second run, or remove
the queue line first. Never patch in place and hope the runner has not popped it. When in doubt compare
`ls --time-style=+%H:%M:%S` on the script against the runner's start line, which is what caught this.

## LESSON 49 — a cross-run control must compare like units, and a section often publishes two

§1811's pred_d required this run to reproduce §1805's L10 figure. §1805 published it both ways — as a
percentage-point delta (+13.86pp) and as a fraction of the gap (53.8%). I stored the pp value and
compared it against the fraction, so the control asked whether 0.53774 equals 0.1386 and answered no.
The data reproduced §1805 to four decimal places (|d − published| = 0.00004 / 0.00001 / 0.00004); the
predicate was measuring the wrong quantity.

The failure mode is worse than a wasted bar. A cross-run control exists to catch a changed definition
(LESSON 42), and one that fails for its own reasons is indistinguishable at a glance from one that
caught something real — I had to compute all four conjuncts separately to know which had fired. Three
of them held to five decimals and said so only under inspection.

**How to apply.** When quoting a published figure into a predicate, quote the UNITS with it in the
constant's name or a comment (`S1805_L10_PP` not `S1805_L10`), and compute the comparison quantity in
those units at the point of use. When a control fires, decompose it conjunct by conjunct BEFORE reading
anything into the result — a conjunction reports one boolean for many claims, which is [[lesson-44]] in
another costume. Related: [[lesson-36]] (never compare a rounded number against an unrounded one).

## LESSON 50 — a cost model must charge for capacity the object can actually use

§1813 priced a rank-64 embedding->row map at 5.308M reals and concluded that at table rank 1 the map was
94.3% of the program, so the thread had been "optimising the minority of the bill". Both statements were
artifacts of the price. The map is a truncation of `Ws = A^-1 Ecov^T tables`, whose rank is bounded by
the table's: at table rank 1, `rank(Ws) <= 2`, and every map rank from 8 to 256 is the SAME MATRIX.
I was charging 32x for capacity the algebra forbids the object from holding. Priced at the rank it can
carry, the map is 26-34% of the bill at every table rank and the rank-1 program costs 0.485M, not
5.628M -- **11.6x less than I published one section earlier**.

The error was invisible in the regime the accounting was built for. At table rank 64 the cap is 65, so
charging rank 64 is exactly right, and every §1754-§1787 figure is unaffected. It only bites once the
table rank falls below the map rank -- a regime the cost model was written before anyone entered.

**How to apply.** When a cost formula has two independently-parameterised terms, check whether the
parameters are actually independent before sweeping them. Here one line of algebra -- what is the rank
of the thing being truncated? -- was available at any point and would have caught it. Symptom to watch
for: **arms that come out bit-identical across a parameter sweep.** That is not noise and not luck; it
means the parameter is not reaching the object, either because of a bug or, as here, because it cannot.
Related: [[lesson-44]] (emit the discriminating quantity) -- the identity of the t1 arms across four map
ranks WAS the discriminating quantity, and it was visible in the log before the run finished.

## LESSON 51 — I rebuilt a metric whose defect I had already written down

§1815's pred_c scored "CE nats bought per million reals against the fully-tabled program". Every arm in
that run is a truncation of the fully-tabled program, so every numerator was negative and the argmax
selected the least-negative arm -- the most expensive one -- by construction. The result is void.

§1787 had the identical defect and I described it in the ledger at the time: *"measured improvement over
the best arm, so every cheaper arm had a negative numerator and the optimum landed on `full` by
default."* Writing the sentence did not stop me writing the metric again eight sections later, because
the check I run before a run is "does this predicate have a margin and a two-sided reading" (LESSONS 40,
31, 39) and this one had both. What it lacked was a numerator that could be positive for the arms it was
ranking.

**How to apply.** Before registering any RATIO, evaluate its sign for the arms you expect to see. If the
numerator is negative for all of them, the ratio ranks by denominator alone and the prediction is about
cost, not about what you meant. Then ask whether the ratio is needed at all: for a two-axis trade-off a
**Pareto frontier requires no baseline** and cannot be broken this way, which is why §1811's dominance
result survived while its efficiency reading did not. Related: [[lesson-35]] (a guard on a denominator
manufactures a number) -- same family, opposite end of the fraction. The deeper point is that a lesson
recorded in a ledger section is not a check; only something that runs is a check.

## LESSON 52 — a degraded-state test must reproduce the defect's STRUCTURE, not just its name

After §1815's `ce_dominance_check` died with `cannot access free variable 'm'` -- a Pareto marker named
`m` inside `main()` shadowing the module-level model that the nested `build()` closes over -- I added a
check for that class to `ops/gate.py` and wrote a test file to prove it fires.

**The first test file passed, and the check was a no-op.** I had written the reader as a MODULE-LEVEL
function while the real defect had it NESTED inside the assigning function. The gate's check walks for
nested `FunctionDef`s, so it correctly saw nothing to flag; my test simply did not contain the bug it
claimed to contain. Had I stopped at "degraded test ran, gate said PASS, good" I would have committed
decoration and believed the class was covered.

Rewriting the test with `build()` nested inside `main()` made it flag immediately, with the exact
message, and `python3` on the same file confirmed the identical NameError. The fixed variant passes and
runs. Four real scripts still pass, so it does not flood.

**How to apply.** LESSON D says test a watcher in both directions; this is the sharper version: the
degraded fixture must be a MINIMAL REPRODUCTION of the actual failure, and the way to confirm that is to
run the real interpreter/tool on the fixture and see the real error. If the fixture does not fail
WITHOUT your check, it cannot demonstrate anything WITH it. A degraded-state test that passes is
indistinguishable from a working check and is the more dangerous of the two.

## LESSON 53 — a cross-run control must name the OBJECT, not just the number and its units

§1820's pred_d compared a run built with rank-64 tables against §1789's published figures for the
FULL-RANK program. The baseline it measured, 0.1288 / 0.1349 / 0.1289, reproduces §1786's rank-64
values to four decimal places -- |d| = 0.0000 -- and misses §1789's by 0.0067-0.0076 against a 0.001
bar. The data were right and the predicate asked the wrong question.

This is the second such failure in one session. §1811's pred_d compared a percentage-point delta
against a gap FRACTION (LESSON 49, units). This one compared the right units of the wrong OBJECT. Both
times three of four conjuncts held to five decimals and I had to decompose the conjunction by hand to
learn which had fired -- and both times the honest write-up had to say "the control failed and the
science is unaffected", which is a sentence that should make anyone suspicious and therefore must be
backed by the decomposition rather than asserted.

**How to apply.** A quoted constant needs three things in its NAME: the section, the units, and the
object -- `S1786_RANK64_TOP1_PP`, not `S1789_PROG`. Before registering, ask "which build produced this
number, and is my run that build?" Here the answer was visible in one line of the script: the arm is
constructed by `build(NFULL, 64, 64)`, so only rank-64 constants can be quoted against it. Related:
[[lesson-49]], [[lesson-42]] -- the cross-run control is still the only thing that catches a changed
definition, which is exactly why mis-specifying it is expensive.

## LESSON 54 — building a script by editing a previous one leaves duplicate constants, and the later one silently wins

`ops/bottom_up_gain_rescue` set `DEPTHS = (0, 3, 5)` for §1806's pathological depths. Eight lines
later an inherited `DEPTHS = (-1, 7, 10, 13)` from the previous script in the lineage silently
overrode it. The run executed the WRONG DEPTHS for three minutes of GPU time and then died on a
`KeyError: 'B3_first'` -- an arm that was never created, because the loop had built B7/B10/B13 instead.

The defect is invisible when your intended value happens to come SECOND and fatal when it comes FIRST,
which is why it survives: it is silently correct most of the time. **Measured across the whole
directory: 12 of 99 ops scripts carry a duplicate module-level constant.** Every one is a true
positive and every one is a single edit away from doing the wrong thing quietly.

Added to `ops/gate.py`, tested both directions per LESSON D and with a fixture that actually contains
the defect per LESSON 52: it names `DEPTHS` on the broken file, says nothing on the same file with the
duplicate line deleted, and Python confirms on the fixture that the later value is what runs. It fires
on 12% of existing scripts, which is not a flood -- those twelve really are defective.

**How to apply.** When a script is derived from another by editing, the constants block is where stale
definitions accumulate, because new ones get PREPENDED and old ones are easy to miss. Grep `^NAME =`
before running, or let the gate do it. And if a duplicate is found in something already run, check
which value won before assuming the published result is wrong -- usually the intended one was second
and the output stands.

## LESSON 55 — a mean over components cannot show an anti-aligned one, and a printed sample chose the wrong four

§1825 measured cosine between each live layer's write and the live model's, across thirteen layers, and
I reported the **mean (+0.7698)** plus **four sampled layers (L4, L8, L12, L17)**. Both hid the finding.
The full artifact has **L9 at −0.134 at B3 and −0.628 at B5** -- the only layer at any depth below
+0.50, and it is *anti-aligned*. Every other layer sits between +0.706 and +0.929.

Codex read my committed artifact and reported the L9 reversals before I did. My write-up said
"directions are ~77% aligned"; the true statement is "directions are preserved everywhere except L9,
which reverses, and reverses harder the deeper the prefix". Same data, and the second sentence is the
science.

The mean was the wrong summary for a quantity that can change SIGN: averaging +0.8 over twelve members
with one at −0.6 gives ~0.7 and reads as uniform mildness. The four printed layers were chosen for
even spacing, which is a display convention, not a search.

**How to apply.** For any per-component quantity, print the **extremes** -- min, max, and which
component -- not a sample and a mean. If the quantity is signed, state explicitly whether any component
crossed zero, because that is a categorical event a central tendency cannot represent. And when a peer
reads your artifact and finds something you did not report, the artifact was right and the write-up was
lazy: fix the write-up, credit the catch, and check what else the same summary was hiding. Related:
[[lesson-44]] -- emit the discriminating quantity, of which "which component is worst" is the simplest
possible case.

## LESSONS 56 — the docstring gets rewritten and the BANNER does not, so the log header names the wrong experiment

Building a script by editing a predecessor (the house pattern, and the right one) rewrites the header
comment, the predictions and the output path — and leaves the runtime `print()` banner untouched,
because nothing reads it during development. So `ops/cross_position_influence.py` ran correctly, wrote
the right JSON, and opened its log with:

```
SECOND MOMENT | across-position dispersion of corrected writes, depths (0, 3, 5) | ...
```

**The banner is the first line a reader sees and the line a write-up is quoted from.** §1828 was written
from that log. Nothing was mis-stated because I had the run fresh in context — but a stale banner is a
mis-attribution waiting for the reader who does not.

This is the same failure family as LESSONS 54 (a duplicate constant the later assignment silently wins)
and LESSONS 53 (name the OBJECT, not just number and units): **editing a predecessor propagates whatever
you did not think to look at.** It had propagated two generations here, second_moment ->
cross_position_influence -> bottom_up_depth_curve, and would have gone further.

**Measured before flagging, per PRE-FLIGHT B.** Auditing all 107 ops scripts for a house-convention
banner (`TITLE IN CAPS | detail`) sharing no word with its filename: **16 of 101 scripts that have one,
all 16 true positives on inspection.** The oldest chains are six scripts still printing `CONTEXT-FREE
FRONTIER, THIRD ROLE` and five printing `ACCURACY BY TARGET FREQUENCY`. `OUT` paths, by contrast, were
correct in **107 of 107** — the artifact naming was never the problem, only the human-facing header.

**Gate check added** (`ops/gate.py`), tested in both directions per PRE-FLIGHT D: FAILs the three known
stale scripts, PASSes correct ones, and stays silent on scripts with no banner. It immediately caught
`cross_position_influence.py`, which I had listed as a should-PASS when writing the test — the check
was right and I was wrong about my own script, which is the best evidence it earns its place.

**How to apply:** when a script is built by editing another, change the banner in the same edit as the
docstring. The gate now enforces it before queueing.

## LESSONS 57 — a share-of-the-total bar cannot separate DOMINANCE from REDUNDANCY

§1830 asked which site of layer 1 buys the 38.9pp drop, and registered pred_a as: *the larger of the two
single-site drops is at least 70% of the joint drop.* It passed at **99.5%**. It should not have been
counted as evidence for anything.

The result was that **both** sites individually cost ~100% of the joint drop — attn1 +37.4pp, mlp1
+38.7pp, joint +38.9pp. That is redundancy, and it is the opposite of the "one site dominates" the bar
was written to detect. **A threshold on `max(a, b) / joint` is satisfied identically by "a is everything
and b is nothing" and by "a and b are each everything".** The statistic has no power to distinguish them
because it never looks at the smaller value.

The prediction that carried the finding was pred_c, which asked about **additivity**: `(a + b) / joint`,
which came out **1.96x** against 2.00x for exact redundancy. That single ratio separates all three
outcomes — ~1.0x additive, ~2.0x redundant, >2.0x super-additive — where the share bar separates none.

This is a sharper instance of LESSONS 44 (a binary predicate cannot separate three outcomes) and of
LESSONS 48 (a marginal along one ordering is not an attribution to a component): the failure is not that
the bar was too loose, it is that **the quantity was the wrong one**, and no threshold on it would have
helped.

**How to apply:** when a prediction is about attributing a total to parts, register the ratio
`sum(parts) / total`, not `max(part) / total`. Ask the discriminating question first — *are these
alternatives, or contributors?* — and only then pick a bar. And when a prediction passes on a bar that
could not have failed for the reason you cared about, say so in the write-up instead of banking the pass.

## LESSONS 58 — a MONOTONICITY prediction must compare NESTED sets, or it is measuring nothing

§1831 registered pred_c as: *no single-site arm recovers less than B1's 25.9%, i.e. compiling one site is
never worse than compiling two.* `B0+attn2` came in at **21.9%**, below B1, and the prediction failed. Its
registered failure branch said this would mean "adding a compiled site can HELP, and recovery is not
monotone in the compiled set."

**It means nothing of the kind.** B1 is `{attn0, mlp0, attn1, mlp1}` and `B0+attn2` is
`{attn0, mlp0, attn2}`. **Neither contains the other.** Two non-nested sets can be ordered any way at all
without saying a word about monotonicity, which is a statement about a set and its supersets.

The run did contain valid nested chains, and both are monotone: B0 64.8% ⊃ B0+attn1 27.4% ⊃ B1 25.9%,
and B0 64.8% ⊃ B0+mlp1 26.1% ⊃ B1 25.9%. So the evidence available pointed the *opposite* way from what
the failed prediction appeared to say, and only reading the set membership showed it.

This is the same family as LESSONS 49 and 53 (a cross-run control must compare like units, and must name
the OBJECT) but the defect is structural rather than dimensional: the units matched, the numbers were
correct, and the *relation* being asserted did not exist between those two arms.

**How to apply:** before registering any prediction of the form "X is never worse than Y", write out both
compiled sets and check that one literally contains the other. If they do not nest, either pick arms that
do, or state the prediction as the ordering it actually is and drop the monotonicity language. A bar whose
failure branch does not follow from its own failure is worse than no bar, because it will be quoted.

## LESSONS 59 — the tail indexed a label suffix the arm loop no longer produced, and threw away a finished run

`ops/mlp5_channel_concentration.py` ran **every one of its fifteen arms**, printed all three roles for
each, and then died at the reporting step:

```
  KeyError: 'B0_seq'
```

Its predecessor called `run_g` **twice per arm**, with labels `f'{name}_raw'` and `f'{name}_seq'`. This
script's arm loop calls it **once** per arm with a bare label, because every arm here uses the same gain
treatment and varies only in which mlp5 channels are substituted. I changed the loop and left the tail
indexing `frac[PICK_ROLE][f'{k2}_seq']`. Seven minutes of GPU work completed and was discarded at the
last step.

Nothing was lost scientifically — the per-arm top-1 for all three roles is in the log and the run was
re-scored after a two-line fix — but the failure mode is worth a check because **it fails LAST**. A
KeyError in the tail is invisible to every fast test: the script imports, parses, gates, and runs
correctly for its entire expensive phase before hitting it.

This is the third member of the family LESSONS 54 (duplicate constants) and 56 (stale banner) belong to:
**editing a predecessor propagates whatever you did not think to look at**, and the reporting block is
the easiest thing not to look at because it is at the bottom of the file.

**Gate check added** (`ops/gate.py`), tested in both directions and measured for flooding: it FAILs the
pre-fix file naming the exact suffix and the labels actually produced, PASSes the fixed one, and flags
**0 of 110** existing ops scripts. It collects every literal and f-string-suffix label passed to
`run_g()` and fails if the results dict is indexed with a suffix (`_raw`, `_seq`, `_global`, `_matched`)
that no call produces.

**How to apply:** when the arm loop's labelling changes, re-read the reporting block in the same edit.
The gate now enforces the specific case; the general habit is that a script's last block deserves the
same attention as its first, precisely because nothing reaches it until everything expensive is done.

## LESSONS 60 — `open(p, 'w').write(HDR + open(p).read())` silently discards the file

Prepending a header to an assembled script, I wrote:

```python
open(p, 'w').write(HDR + open(p).read())
```

Python evaluates the call target before the argument, so **`open(p, 'w')` truncates the file to zero
length first**, and the `open(p).read()` inside the argument then returns `''`. The result is a file
containing the header and nothing else. No exception, no warning — the write "succeeds".

It cost one rebuild rather than a run, because the gate caught it immediately with a finding that read
like a scoring error rather than a file error:

```
expected at least 3 pred_* keys, found 0: []
```

That is the check doing its job for a reason it was never written for. The lesson is not really about
this idiom, which is easy enough to avoid once seen; it is that **a "wrote the file" step can destroy
content while reporting success**, so the thing to verify after assembling a file is its SIZE or a
count of something it must contain, not the absence of a traceback. The rebuild now prints
`len(full)` and the number of `pred_` keys for exactly that reason.

**How to apply:** read first, write second, as separate statements — `t = open(p).read()` then
`open(p, 'w').write(HDR + t)`. After any programmatic assembly, print a length or a content count and
look at it. And keep the source pieces on disk (the body went to the scratchpad before assembly), so a
destroyed intermediate is a one-command rebuild instead of a retype.

## LESSONS 61 — separate shell commands do not inherit each other's failure, so a failed build still queues

Twice today I ran, as one tool call:

```bash
python3 - <<'PY'   ... assemble ops/foo.py ...   PY
bash ops/gpu_free.sh && echo /abs/path/ops/foo.py >> queue.txt
```

The assembly raised an `AssertionError` and wrote nothing. The **next** command is independent, so
`gpu_free.sh` succeeded and the path went into the queue anyway. The runner then popped a path to a file
that did not exist and dropped it silently. No GPU was wasted either time, but the lane sat empty when I
believed it was fed — which is the one thing the wake prompt asks me not to let happen.

The `&&` I did write only chained the GPU check to the append. **The build was never in the chain at
all**, and it is the step most likely to fail, because it is the one I am actively editing.

**Fix: `ops/enqueue.sh`**, which puts all four preconditions behind a single exit code — the file exists,
it parses, `ops/gate.py` passes, and the GPU is free — and refuses with a reason otherwise. Tested in
both directions: it refuses a nonexistent path with `REFUSED: no such file` and exit 1, and queues a real
gated script with exit 0. It replaces the `gpu_free.sh && echo >> queue.txt` idiom everywhere.

This is the same family as LESSONS 41 (a pipeline's exit status is the last command's, so a failed push
printed OK) and LESSONS 60 (`open(p,'w')` truncating before the read evaluates): **the shell and Python
both let a failed step be followed by a successful-looking one.** The general habit is that any check
worth doing belongs in the same expression as the action it guards, not next to it.

## LESSONS 53 — ADDENDUM (third instance): a wrong PUBLISHED ANCHOR is invisible to the gate

§1848's pred_d failed because I checked an all-position CE against §1768's published **6.57512** while
running a **full-rank** program that produces **6.01167**. §1768's figure is the rank-64 table with
rank-128 corrections; mine was the settled full-rank build. Same pipeline, same roles, same population —
**different object**, and a 0.56-nat difference read as a control failure when it was a construction
difference.

This is LESSON 53 for the third time (§1811's pp-versus-fraction, §1820's full-rank constants at a
rank-64 build, now this), and the recurrence is the point: **the gate cannot catch it.** A wrong anchor
is a syntactically perfect float. The duplicate-constant, stale-banner and label-suffix checks all work
because the defect is structural; this one is semantic.

What saved it was a *second* anchor pointing the other way: the same run reproduced §1811's published
settled full-rank figures **6.01167 / 5.98477 / 6.00165 on all three roles**, exactly. Two anchors
disagreeing is what localised the error to the anchor rather than the run.

**How to apply, as a habit rather than a check:** when registering a cross-run constant, write the BUILD
that produced it next to the number — rank, fallback, scored population, and the arm — not just the
section number. `S1768_PROG` should have been `S1768_PROG_RANK64_CORR128`. And prefer **two** anchors
from different sections where the run overlaps both: a single anchor can only tell you something is
wrong, never which side of the comparison it is on.

## LESSONS 62 — an OOM whose byte counts do not move means your fix freed nothing

`ops/rank_to_ceiling.py` OOMed four times. Three times I "fixed" the memory — removed an unneeded row
bank, freed the previous rank bank, released the full-rank bank before building the next — and three
times the traceback came back with **exactly** the same numbers:

```
  19.59 GiB allocated by PyTorch, 10.88 GiB reserved but unallocated, 247.94 MiB free
```

**Identical byte counts across a change mean the change had no effect on what is resident.** I read each
recurrence as "still too tight" and made the next allocation smaller, when it was saying "you did not
free the thing you think you freed".

The cause: `hks = [(st, row_hook(bank[st])) for st in sites]`. `row_hook` closes over the individual
tensors, so `bank = None` drops the dict and leaves every tensor referenced by a live closure in `hks`.
Clearing both names fixed it on the first try.

**How to apply:** when an OOM repeats with the same numbers, stop shrinking allocations and find what
still holds a reference — closures, lists of hooks, and captured loop variables are the usual culprits,
because none of them look like a tensor. And in this codebase specifically, **a `row_hook` is a handle on
an 8.3 GiB bank**: clearing the dict it came from is not clearing the bank.

## LESSONS 63 — I changed the ladder and the docstring, and left the predicate block scoring the old run

`ops/frontier_top1_16110.py` was built by editing `frontier_knee.py`. I changed `TRANKS`, the rank list,
the cost table and — carefully — wrote four **new** registered predictions into the docstring, about
whether CE and top-1 rank the builds the same way.

The code still computed `frontier_knee`'s predicates. The log printed "the curve BENDS EARLY", "NOTHING
in 384..1024 dominates full rank" and "the ladder stays MONOTONE in rank": the *previous* script's
questions, answered correctly, under a docstring registering three different ones. **The printed booleans
had nothing to do with the registered predictions.**

Nothing was lost — the registered predictions were all computable from the recorded curve and I scored
them by hand — but the failure mode is bad: a reader comparing the docstring to the log sees four
`pred_*` lines and four booleans and has no reason to suspect they are about different things. **It is
worse than a crash, because it produces a plausible answer to the wrong question.**

This is the same family as LESSONS 56 (stale banner), 59 (stale label suffix) and 61 (the failed build
that still queued): **editing a predecessor propagates whatever you did not think to look at, and the
reporting block is the part you do not look at.** It is the third time this evening.

**How to apply:** the docstring and the predicate block are one edit, never two. When the registered
questions change, rewrite `pa`/`pb`/`pc` in the same pass — and before queueing, read the `print` lines
that report them against the docstring lines that register them. They should say the same thing in the
same words; if they do not, one of them is from the last script.

## LESSONS 63 — ADDENDUM: the audit, and which direction actually matters

The check written for LESSONS 63 was run across all 132 ops scripts and found **14 inherited
registered-prediction lines in 9 scripts** (§1857). Two things worth carrying beyond the specific fix:

**The two directions are not equally dangerous.** A stale *docstring* over fresh code is a record defect:
the bars were still fixed before the run, in the predicate expressions and in the print strings, so
nothing floated. A stale *code block* under a fresh docstring — §1856 — answers a question nobody asked
and looks entirely normal doing it. When triaging, check which way round it is first.

**The unrestricted check was too noisy to use.** Requiring only that key and docstring share a content
word flagged 35 of 341 pairs, mostly benign paraphrases (`single_basin` for "BOTH STARTS CONVERGE").
Adding "and the docstring line appears verbatim in another ops script" cut it to 14, all true positives.
**The signal was never the mismatch; it was the inheritance.** A check aimed at the symptom floods; the
same check aimed at the mechanism does not.

## LESSONS 64 — a results key that only exists conditionally, and a regression fixture that was a no-op

`ops/iso_cost_rank.py` built all four rank tables, then died in its reporting block:

```
  KeyError: 'full'
```

The rank-sweep lineage keys its results dict by `str(rank)` with `None -> 'full'`. This script dropped
full rank from `TRANKS` and kept the inherited `curve['full']` lookup. **Fourth tail-inheritance failure**
in this codebase (LESSONS 56 banner, 59 label suffix, 63 predicate block, now this), and the first that
is **statically decidable**: if `TRANKS` contains no `None`, `curve['full']` cannot resolve. Gate check
added; it flags 0 of 138 existing scripts.

**The more useful half of this is the fixture.** I built the must-fail case with a `sed` substitution
whose pattern did not match — the replacement text contained quotes and brackets — so the "broken" file
was **identical to the fixed one**, and the new check "passed" its must-fail test by never being
exercised. I only caught it because PRE-FLIGHT D says to test a watcher in the degraded direction and I
printed the intermediate: `curve['full'] found: False`.

**This is LESSONS 52 repeated exactly** — there too my first regression fixture was a no-op and the check
looked fine. The pattern is now twice-observed and worth stating as a rule:

**How to apply:** build regression fixtures with an explicit assertion that the defect is present
(`assert "curve['full']" in broken`), not with a text substitution you assume worked. A fixture that
silently fails to introduce the bug makes a broken check indistinguishable from a working one, and the
whole point of the must-fail test is that it can only be trusted if the fixture is verified first.

## LESSONS 65 — I refined one gate check four times and it went SILENT twice; both silences were the fixture or the comments

Adding the generalised LESSONS 64 check (`int(r)` on a LADDER holding non-numeric entries) took four
attempts, and the failure modes are worth more than the check:

1. **Fired on its own explanatory comment.** The check searched the whole file, and my comment said
   "rcost() does int(r)". Fixed by searching `_code` with comment lines stripped — **the identical fix I
   had already applied to the `curve['full']` check an hour earlier and did not carry over.**
2. **Went silent because LADDER was built from a variable.** `LADDER = list(ARMS2)` has no string
   literals, so the check read nothing and passed a genuinely broken file. Fixed by resolving one hop to
   the referenced constant.
3. **Went silent again on a "guarded" heuristic.** I added a skip for files containing `== 'full'`,
   reasoning that a guard makes it safe; the broken fixture also contained that string in inherited code,
   so the check stopped firing on the very file it was written for. Removed the heuristic.
4. **A fixture that did not contain the defect** — the same LESSONS 52 error as earlier the same night,
   caught only because I printed whether the pattern was present rather than trusting the edit.

**Two of four failures were the check going silent, and PRE-FLIGHT D exists precisely because that is the
direction you do not notice.** A check that fires wrongly annoys you into fixing it; a check that stops
firing looks like success.

**How to apply:** (a) any pattern check over source must read code with comments stripped — make that the
default, not a per-check fix; (b) resolve one level of indirection before concluding a literal list is
empty, or the check silently describes nothing; (c) do not add "safe case" skips to a check without
re-running the must-fail fixture, because a skip is the easiest way to silence it; (d) assert the fixture
contains the defect before trusting either outcome. The gate ended at 2 of 150 flags, both explained —
guarded, unreachable `int(r)` in scripts that already ran — which is the state to record rather than
tune away.

## LESSON 66 — a replacement block's trailing indentation re-indents the line that FOLLOWS it

`stream_input_closure.py` was built by string surgery on `deployable_stream_map.py`. The replaced slice
ended just before `        del tables, Ecov, Eunc, A`, and my replacement string ended with a newline and
eight spaces. Eight plus eight is sixteen: the retained `del` landed **inside the `for st in sites:` loop
I had just inserted**. It parsed. `ops/gate.py` passed it. `ops/enqueue.sh` queued it. It died **271
seconds in**, on the second iteration, with `UnboundLocalError: cannot access local variable 'tables'` —
after both other arms had run and returned their covered CE at exactly 0.00000.

This is the second splice failure in one session. The first (`deployable_stream_rank1024.py`) produced a
`SyntaxError` and was caught for free by `enqueue.sh`. **This one produced valid Python, so every
syntactic guard I own was blind to it.** The difference is entirely luck about where the collision landed.

**The rule:** when a replacement's boundary abuts a retained line, the retained line's own indentation is
still in the file. End the replacement at a newline with **no trailing whitespace**, or include the
retained line in the replacement and own its indentation explicitly. Do not rely on the two agreeing.

**The check** (now in `gate.py`): `del X` inside a `for`/`while` whose body never assigns `X`. A
loop-invariant delete-in-loop is always a bug, and it is the exact fingerprint an indentation splice
leaves. Building it took three drafts, and **LESSON D's two directions caught both wrong ones**:

```
  draft 1  bare ast.Name targets only          FIRES on the bug   FLOODS: 28 of 156 working scripts
  draft 2  + tuple-unpacking targets            FIRES              2 of 156 -- rank_crossover{,_v2}
  draft 3  + the loop's OWN target              FIRES              0 of 156
```

Draft 1 missed `bank, seen, n = build(...)`. Draft 2 missed `for name, sc in (...): ... del sc`, where
the loop target rebinds every iteration — both flagged scripts had **exit=0** in `_completed.txt`, which
is how I knew they were false positives rather than a backlog. **A flood is a failed check, not a
finding**: I only trusted the third draft because it fires on the reintroduced bug AND is silent on all
156 scripts that actually ran.

## LESSON 67 — a gate refinement that reaches FORWARD for a variable goes silent on every script at once

Refining the empty-literal check to spare live accumulators (`ITER_DELTA = []` filled by `.append()`), I
wrote the rescue clause at line 171 against `_code` — which `gate()` does not define until line **281**.
`UnboundLocalError` on **145 of 157 scripts**, and because my two-direction test greps stdout for the
finding, the crash read as **"silent"** in BOTH directions. Two drafts in a row (`code`, then `_code`)
failed the same way and printed the same reassuring word.

**LESSON 65 said a refinement can go silent. This says the silence can be a CRASH wearing a passing
test's clothes.** A check that greps for the presence of a finding cannot tell "correctly quiet" from
"the tool died". The fix is one line in the harness — assert the tool didn't traceback and count it:

```
  crash = 'Traceback' in r.stderr
  print(f'-> {"CRASH" if crash else ("FIRES" if TOKEN in r.stdout else "silent")}')
```

That third state is what turned two silent failures into two visible ones. **The flood test needs it
too**: "0 of 157 flagged" looked like a clean result and was really 145 crashes.

The substantive fix was to hoist the comment-stripper into a module-level `_nocomment(src)` instead of
reaching forward for a local. **Reaching forward for a variable inside a long function is the same class
of error as LESSON 66's indentation splice** — both are edits that assume a context the edit site does
not have. When patching into the middle of a function, check what is actually bound *there*.

## LESSON 68 — a convergence bar must be absolute, not relative to your own first iterate

§1879's pred_d certified that the map iteration "CONVERGES — the last relative map change is smaller than
the first." It read **22.63370 → 5.43549 → 1.85551** and returned **True**. Monotonically falling, and a
relative change of **1.86 means the map turns over completely every pass.** Nothing converged.

The clause is true as written, so it was scored True — and it certified nothing. **A bar anchored to your
own first iterate cannot fail as long as the sequence decreases at all**, no matter where it decreases
*to*. The bar had to be absolute: a relative change below some fraction of 1.0.

This is LESSON E's shape one level up. E says never use a fixed absolute tolerance on a spectrum — scale
it. **68 says the opposite error exists**: a purely relative bar, scaled to a quantity you chose rather
than to the thing being certified, is unfalsifiable. The test for both is the same question — *what
reading would make this predicate FALSE?* If the answer is "the sequence increasing", and the sequence
was always going to decrease, the predicate is decoration.

The damage was bounded because pred_a and pred_b were direct measurements and failed honestly. But the
run's headline row is **iterate 3 of a non-converging sequence**, and the ledger now says so rather than
citing it as a fixed point.

## LESSON 69 — the gate checks that a predicate RUNS, not that it MEANS what the docstring says

Two bad predicates in three sections, both gate-clean, both scored:

- **§1879 pred_d** certified convergence as *"the last relative map change is smaller than the first"* and
  passed on **22.63 → 5.44 → 1.86** — a map still turning over completely every iteration. Unfalsifiable
  as long as the sequence decreases at all (LESSON 68).
- **§1884 pred_a/pred_c** used `acc_live * acc_prog` as the chance baseline for top-1 **agreement**. That
  is the probability both are RIGHT, not the probability they AGREE. It assumes disagreeing predictors
  scatter wrong answers uniformly over 50,257 tokens; both of these concentrate on frequent tokens, so
  the null is far too low — and the inflation is worst where accuracy is lowest, which manufactured a
  **97x** "enrichment" on the rare bucket and a spurious inversion.

**The shared tell: neither bar was ever evaluated on a case whose answer I already knew.** Both would
have been caught in under a minute of arithmetic before the run — 1.86 is obviously not converged, and a
97x enrichment is obviously implausible for two predictors of the same text. I did that arithmetic *after*
reading the output, both times.

**The rule: before registering a bar, evaluate the predicate on one hand-made case where you know the
verdict, and check it gives that verdict.** For a convergence bar, feed it a sequence that plainly has not
converged and confirm it says FALSE. For a chance baseline, compute it on two predictors you know to be
independent and confirm the ratio is ~1. This is LESSON D's two-direction test applied to *predicates*
rather than to watchers — the same discipline, one level up, and I had not been applying it there.

`ops/gate.py` cannot catch either: both are semantically wrong and syntactically perfect. **This one is
mine to do by hand, and the cost of skipping it is a run plus a struck result.**

## LESSON 70 — apply a guard to every predicate it protects, and print the n in the predicate line

§1889 split the fallback arm into four rarity bands. Three were empty or near-empty, because the axis I
chose (the current token's fit-row count) is collinear with the coverage definition itself — uncovered
tokens have fit-count 0 in **99.9%** of cases. The run could not answer its question.

That was a design error, and it was compounded by a smaller one that is more general. **I wrote the guard
and then used it in only two of the three places it was needed.** `live_bands` required `n >= 100`
exactly because empty bands give meaningless enrichments; pred_b and pred_c used it, pred_a did not, and
pred_a returned **True** by observing that an empty band (0.00x, n = 0) scores lower than a populated one.

**Two rules from this:**

1. **A guard belongs to the quantity, not to the predicate.** If a statistic is only meaningful above some
   n, filter it once where it is computed, so no downstream predicate can reach the unguarded version.
2. **Print the n in the predicate line.** The output read `COMMONEST band tracks WORSE than the rarest ->
   True  skip7000 100-1000000000 0.00x vs 0-0 2.92x`. Everything needed to see the bug is in the table
   two lines above — and the predicate line, which is what I read first, showed a clean pass. A `0.00x`
   should be impossible to print without the `n=0` beside it.

This is the third predicate failure in eleven sections (LESSON 68's unfalsifiable bar, LESSON 69's
mis-specified null, this). All three passed `ops/gate.py`. The pattern is not carelessness in coding — it
is that I keep checking whether a predicate RUNS and not whether it can be satisfied by a degenerate
case. **The one-minute habit that catches all three: before the run, ask what the WORST possible data
would print, and check the predicate says FALSE for it.**

## LESSON 71 — two instruments that score different POPULATIONS will contradict each other forever

Six ledger sections (§1898-§1903) went round a loop because §1898 counted prediction changes at **all**
scored positions while §1899 and §1901 measured streams and outputs at **covered current tokens only**.
Every individual measurement was correct. They simply were not about the same thing, and comparing them
produced an apparent impossibility that survived four rounds of hypothesising — the `v1` residual, float
argmax flips, bfloat16, non-determinism — each refuted by a run.

**The population restriction was MINE and I had written it down.** §1899 and §1901 both contain
`msk = seen[flat]`, and both registered pred_d as *"the comparison is run on COVERED current tokens only,
since an uncovered token has no table row."* I then compared their results against an all-position count
as though the scopes matched. **Codex found it from the source in the time it took me to build a fifth
measurement**, and localised 99.24 / 99.41 / 100% of the changes to uncovered tokens.

**This is the fourth population-dependence failure this session:**
- §1882 lost a launch to a live-CE anchor measured on the 5,419-type covered set, used at 16,110.
- §1889 was a null run because the rarity axis was collinear with the coverage definition.
- §1890 refuted §1888's compositional guess about a coverage-driven sign.
- §1903, here.

**The rule: when two numbers disagree, check that they were computed over the same positions BEFORE
proposing any mechanism.** It is one grep for the mask and it precedes every hypothesis. I have written
three lessons this session about registering bad predicates (68, 69, 70); this one is about comparing
good measurements that were never comparable, which is cheaper to cause and far more expensive to unwind.

**And the corollary that would have caught it in one step:** a quantity's scope belongs in its NAME. Had
§1899 reported `rel_diff_covered` rather than `rel_diff`, the comparison against an all-position count
would have looked wrong on sight.

## LESSON 72 — a bar on |Δ| cannot test a claim about direction, and I wrote two in a row

§1932's pred_a: *"the combined build's kept-fraction in the 125+ bucket is within 2 percentage points of
the deployed design's"*. It passed at 1.20pp — **while the sign flipped from favourable at 16,110 to
unfavourable at 5,419.** The predicate was about magnitude; the finding was the reversal, and the True
told me nothing about it.

§1933's pred_c: *"the map is not the cause: going from the rank-64 to the rank-512 map moves both buckets
by less than 1.5pp"*. It passed at 1.33pp — **while that 1.33pp IS the entire rare-end effect the section
was trying to attribute.** I had written a predicate whose pass was compatible with the map being wholly
responsible for the thing I was asking about.

**Both were framed as causal or directional questions and tested with a two-sided tolerance on an absolute
difference.** A `|Δ| <= t` bar answers "is the change small", never "which way did it go" or "is this the
cause". They are different questions and only one of them was mine.

**The rule: when the registered question contains a direction or an attribution, the predicate must
contain a SIGNED comparison or a contrast against an alternative — not a tolerance.**
- direction → `a < b` on every role, not `abs(a - b) <= t`
- attribution → compare the candidate lever against a lever held fixed, and require the candidate to
  account for most of the effect, not merely to be small

Both sections' *results* survive because I read the tables rather than the booleans. **But two passes in
two consecutive sections carried no information, and if I had trusted the pred lines I would have
published "structure preserved" and "the map is not the cause" — the second of which is false.**

Related: LESSON 68 (a bar relative to your own first iterate is unfalsifiable) and LESSON 69 (calibrate a
bar on a known-answer case first). This is the third distinct way a syntactically fine predicate can fail
to test its own question, and the gate cannot catch any of them.

## LESSON 73 — two files with the same name in two repos, and the push script made the mistake look like success

I appended a board note to `theseus-bench/AGENT_BOARD.md` and pushed. `ops/push_both.sh` reported
**PUSHED, verified 0 commits ahead** for both repos — a completely green result for a note that nobody
would ever read. The live board is `tensor_language/AGENT_BOARD.md`: 906KB, 74 of my own prior notes, the
file Codex writes to and the one the monitor watches. The theseus-bench file is a 4.6KB stub whose last
entry is from 2026-08-27, two days stale. I had posted to the live one 74 times and still got it wrong,
because I reached for the repo I had just been editing (the registry) rather than the repo the board
lives in.

**Why it survived every check I have.** The gate reads scripts, not markdown. `push_both.sh` verifies
that commits reached origin, which they did. Nothing in the loop verifies that a note landed where the
*reader* is. A same-named file in the sibling repo is exactly the shape that defeats a
did-the-write-succeed check: the write succeeds.

**The rule.** Before appending to a shared file that exists in more than one repo, confirm the target by
its *content*, not its name — `wc -c` and `tail` it, and check that the peer's most recent entry is in
it. "The file has the right name and my write succeeded" is not evidence I wrote to the right file. Same
family as LESSON 67 (a change that goes silent everywhere at once) and LESSON 71 (two instruments scoring
different populations): the failure is invisible from inside the thing that failed.

## LESSON 74 — a bucket axis names ONE thing, and I read a second thing into it

§1935 found the map's gain in the 1-4 through 25-124 target buckets and wrote: "the 1-4 through 25-124
buckets are mostly **covered** targets, so the map is doing something beyond serving the uncovered arm."
§1936 measured it: the map changes the top-1 at **exactly 0 of 69,444 covered-input positions.** The
inference was wrong because §1789's instrument buckets on the **TARGET's** frequency and the map is
consulted on the **INPUT's** coverage. A frequent target reached from an uncovered input sits in a high
bucket and is fully exposed to the map. Two different coverage axes, one word.

**Why it got through.** The instrument's own docstring says it plainly — *"The bucket axis is the TARGET,
not the current token"* — and I had copied that docstring forward through four forks. Knowing the axis
was not enough; at the moment I reasoned about a *mechanism* I substituted the axis I cared about for the
axis I had. Same family as LESSON 71 (two instruments scoring different populations contradict each other
forever), but one level worse: here it was one instrument and I misread which population it scored.

**The rule.** When an aggregate surprises me and I reach for a mechanism to explain it, first write down
what the aggregate's axis is keyed on and check that the mechanism acts on the SAME key. If they differ,
the surprise is probably a projection and the fix is a cross-tabulation, not a theory. The
cross-tabulation cost 89 seconds and gave an exact-zero answer.

## LESSON 75 — dead compute behind a confident label survived four forks

Every script in the map lineage (§1933, §1934, §1935, §1936) prints *"the settled fallback: output-NN
neighbour (§1780/§1781)"* and spends several seconds building the neighbour index `nnrow` over all 50,257
types. **`nnrow` is written and never read in any of them** — `program_rows` fills the uncovered rows from
the map alone. The arms were still correct (§1870's settled design *is* map-only for uncovered rows;
230.087M = 224.778M table + 5.308M rank-64 map, against the neighbour's 224.778M), so nothing published
is wrong. But the runtime banner asserted a component that was not in the build, in four consecutive
sections, and I quoted that banner while writing them up.

**Why no check caught it.** The gate flags a `del X` in a loop that never assigns X, and empty literals,
and welded result keys — all *syntactic* smells. An assigned-but-never-read local is legal, cheap-looking,
and its label is in a `print`, which no gate reads. Nothing in the loop compares what a script *says* it
built against what it *uses*.

**The rule.** When forking a script, grep every name in the banner for a second occurrence that READS it.
A name that is only ever written is either dead code or a missing arm — and here it was the missing arm:
the neighbour fallback is the one alternative form this thread has measured, and it was sitting computed
and unused in the exact experiment that needed it.

## LESSON 76 — one instrument chose a component, and nothing re-checked it for 67 sections

§1870 compared two fallback forms in **CE**, the map won by 0.006–0.016 nats, and the map became "the
settled fallback". That choice then rode into §1789's deployed design, §1931's best-known build, and
every section from §1933 to §1936 — all of which varied the map's *rank* and never questioned its *form*.
§1937 scored the same two forms on **top-1** and the ranking **inverts**: the neighbour wins by +2.54 /
+1.62 / +2.47pp on the arm where fallbacks act, +0.61 / +0.41 / +0.60pp pooled over all positions, at
**~0.09M against 5.308M — 59× cheaper and better.**

**Why it survived.** Nothing was wrong. §1870's CE number is right; the arms were right; the deployed
design was assembled correctly from it. The failure is that a *selection* was made under one objective
and then every downstream question was asked *inside* the winner. Sixty-seven sections of careful work on
the map's rank, none of which could ever have surfaced this, because none of them contained the loser.

**The rule.** When a comparison **selects a component** rather than merely reporting a number, the losing
arm has to be re-run the first time the scoring instrument changes — and this thread's instrument changed
at §1788 (top-1), §1789 (buckets) and §1936 (input coverage) without anyone going back. Write the
selecting comparison down as a *decision*, note which instrument decided it, and re-open it when that
instrument stops being the one you report. Cf. LESSON 75: the losing arm was literally still being
computed in every script and thrown away.

## LESSON 77 — CE and top-1 cannot agree about a component that fails catastrophically on a small subset

§1938 measured both instruments on three fallback forms in one build. **The orderings are exact
reversals**: CE ranks map512 > map64 > nn, top-1 ranks nn > map512 > map64. The cause is not noise and not
a near-tie — it is a structural property of the two objectives. On uncovered inputs with an unseen target
the neighbour's CE is **+1.06 to +1.16 nats worse** than the map's, because it emits a real token's
distribution which puts near-zero mass on a token no fit row contains, and CE charges the log of that.
**Top-1 charges at most one position for the same failure**, and the neighbour still scores 5.6–7.7%
there. That bucket is ~5–7% of scored positions and it reverses the entire pooled CE ranking while
leaving the top-1 ranking untouched.

**The rule.** Whenever one arm can fail *unboundedly* on a subset the other cannot — an impossible
target, a zero-mass token, a division by something the other arm never divides by — CE and accuracy will
disagree, and which one is "right" is a question about the objective, not about the model. Report both,
and say which objective a selection was made under. This thread denominated everything in M/nat from
§1747, and that denominator, not any error, is what put the map into the deployed build.

Corollary for predicates: my pred_c here was "the CE penalty is the unseen bucket, ≥0.05 nats". It passed
at **+1.12 nats — twenty times the bar.** A bar that low did not distinguish "localised" from "the entire
effect and then some". When a mechanism predicts *concentration*, the bar should be a share of the total
deficit, not an absolute floor.

## LESSON 78 — I published a headline, wrote the caveat that predicted its death, and then tested it

§1939 claimed §1789's deployed fallback was "strictly dominated on both instruments". Its CE half was
0.9–2.5 milli-nats. I wrote, in the section itself: *"it is a 2–6% effect on the CE axis and I have not
put a paired standard error on it."* Then I published, then I queued the standard error. §1940 came back
with **t = −0.54 / −0.23 / −0.44** — not distinguishable from zero — and at the higher coverage one role
was **significantly worse** (t = +2.59). The headline was retracted a run later.

**The cost.** The paired t is a mean, an sd, and a square root over data the run had **already collected
per position**. Adding it to §1939 would have cost seconds. Instead it cost a published claim, a
retraction, a correction to three places in the record and a registry entry rewrite — and anyone reading
the board between those two sections got a deployment recommendation I could not support.

**The rule.** If I can name the statistic that would falsify a claim, and it is computable from data the
run already has, it is not a caveat — it is a **missing predicate**, and the run does not ship without
it. "I have not put an error bar on it" in a published section is a confession, not a disclosure. The
3/3-in-sign pattern is specifically what a near-zero effect on three roles looks like; sign agreement
across roles is not evidence of size.

## LESSON 79 — a guard written for one action, applied to a different one, caused the largest loss on the box

`ops/enqueue.sh` refused to queue an experiment while the GPU was busy. The guard came from the standing
rule *"never launch onto a busy GPU"* — but **enqueue does not launch anything.** `ops/bqrunner.sh` pops
one line at a time in a single loop and IS the serialization point; appending while a run is in flight is
precisely what a queue is for. The guard's only effect was to make queueing ahead **impossible**, so the
pattern was forced to be: run one, wait, write the next, queue it, run one.

**What it cost, measured.** Over a 48 h window: GPU busy 22.7 h (47%), agent turnaround 17.1 h (36%), and
**long idles 17.4 h (36%)** — a lane sitting empty. Two of the three biggest buckets are the GPU waiting
for me, and the tool was enforcing that it must.

**Why it survived.** The guard was correct-looking, cheap, and had a good comment citing a real lesson
(LESSON 61, about chaining commands whose failure does not propagate). Nobody re-derives the purpose of a
passing check. It only surfaced because I was profiling wall-clock rather than reading code — the numbers
pointed at idle, idle pointed at queue depth, and queue depth pointed at the guard.

**The rule.** A guard states a precondition for a specific ACTION. When the same guard is copied onto a
different action, re-derive it from scratch: what goes wrong if it is absent, *here*? "Don't run two jobs
on one GPU" is a fact about running, not about appending a line to a file. And profile wall-clock, not
just code — a check that is individually correct can still be the system's dominant cost.

## LESSON 80 — the gate only checked names inside functions, and a fork dropped a module-level definition

`ops/blend_meets_allocation.py` was a fork whose header block was replaced wholesale. The replacement
dropped `ALPHA`, but one reference survived in the result payload at module scope: `'alpha': ALPHA`.
Legal syntax, **gate PASS**, and a `NameError` that would have fired **after the entire run**, while
building the argument to `report()` — so all the GPU work and no artifact. I found it by hand-auditing a
script the gate had just passed, which is not a strategy.

**The hole.** The gate's undefined-name check walks **function bodies only**. Every experiment in this
tree does its real work at module scope or in one `main()`, and result payloads are assembled at module
scope, so the single most fork-prone region of every script was the one region unchecked.

**Both directions, and the second one nearly shipped a disaster.** The new check caught the bug on the
first try — and flagged **226 of 227 existing scripts**, all on `__file__`, which the interpreter injects
rather than binding. That is LESSON 67's shape again (a refinement that fires on everything at once). It
was caught only because the first thing I ran it against was the whole corpus rather than the one file I
wrote it for. After allowing the module dunders: **0 of 227 verdicts changed, and the real bug still
fails.**

**The rule.** A new gate check is not done when it catches its motivating case. Run it against every
script in the tree and require a *zero* verdict delta on the ones you have not changed — a check that
changes 226 verdicts is not stricter, it is broken, and one that changes 0 while catching the case is
the only evidence that it is measuring what you think.

## LESSON 81 — I registered a control that was false by construction, and only the run could tell me

§1946's pred_d inherited the clause *"every arm is inert at covered inputs"* from §1936–§1945, where it
was a strong and repeatedly-passing control: those sections varied only the **fallback**, which by §1765
cannot touch a position whose current token has a table entry. §1946 varies the **table rank** as well —
and a truncated table changes covered-input predictions by definition. **The control could not hold. It
failed, while pred_a/b/c passed 3/3.**

**What makes this different from a wrong prediction.** A failed prediction is information. A control that
is false by construction is **noise in the place I put my assurance**: had I not read the clause I would
have concluded the run was broken, and had I quietly dropped it I would have shipped a section whose
controls tested nothing. Thirty-one prior sections passed that clause, which is exactly why it survived
the fork — its track record was evidence about the OLD lineage and none at all about this one.

**The fix, and it is better than the original.** Rewritten two-sided: arms differing only in the fallback
must be **exactly** inert at covered inputs, and arms differing in table rank must **not** be. Asserting
the invariant *and* its negation catches a build that silently stops truncating as well as one that
silently leaks the fallback.

**The rule.** When a lineage gains a new free variable, re-derive every inherited control against it and
ask *what would have to be true for this to fail?* If the answer is "nothing, it cannot fail" the control
is dead weight; if it is "the thing I just changed", it is false by construction. Prefer controls that
assert both a positive and a negative, because those cannot be satisfied by an arm that does nothing.

**Extended §1949 — it happened again, three sections later, and the two-sided form is what caught it.**
§1949's arms all share ONE table spec and differ only in the fallback, so by §1765/§1936 they are
**exactly inert** at covered inputs. I inherited §1947's control, which asserts arms **differ** there —
correct in §1947, where they differed in table *rank*. **The polarity was backwards and pred_d failed
while pred_a/b/c passed 3/3.** So the failure mode is not just "an inherited control that cannot fail";
it is **an inherited control whose polarity is tied to a variable the fork changed**. Two-sided controls
are what make this survivable: a one-sided "arms must differ" would have passed §1947 and §1949 alike by
accident in half the cases, and told me nothing either time.

## LESSON 82 — three variants of a gate check, all measured, all rejected; recording the negative

A fork replacing its arm block leaves behind string literals naming arms that no longer exist:
`res[c][r]['map64']`, `armR['blend_full']`. **Three times on 2026-08-29**, each a `KeyError` *after* the
full run. Name analysis cannot see them — they are `str`, not `Name` — so I tried to build a gate check
and measured three forms against the whole corpus:

1. **any string subscript appearing once in the file** — fires on **218 of 227** scripts (615 hits) on
   ordinary keys like `'rows'`, `'seen'`, `'ce_prog'`. Unusable.
2. **an arm-shaped literal absent from a statically-declared ARMS** — only 2 of 82 flagged, but both are
   false positives from my own extractor, and it **skips every script that builds its labels with a
   comprehension or f-string** — which is exactly the population that drops keys. It did not catch the
   bug it was written for.
3. **a literal subscripting a result container and appearing once in the file** — 35 of 178 scripts
   flagged on legitimate `'baseline'`, `'norms'`, `'self_check'`, and it flags two *valid* keys in the
   very file it was meant to protect.

**Rejected, and the gate reverted.** Writing this down so the next attempt starts from the measurements
rather than the idea. What actually reduced the cost of this bug class was **the cache**: a re-run after
a fix is now ~5 seconds instead of ~270, so the same mistake costs a hundredth of what it did this
morning. Sometimes the right fix for a failure mode is to make failing cheap rather than to prevent it.

**The rule that did generalise** is LESSON 80's: a new gate check is not done when it catches its
motivating case — run it against every script and require a zero verdict delta on the ones you have not
changed. All three variants above died on that test, and each took about two minutes to kill.

## LESSON 83 — reference labels must be NAMES, never string literals. This is the fix; the checker was not.

**Five times on 2026-08-29** a fork renamed or dropped an arm and left a string literal behind pointing at
it — `res[c][r]['map64']`, `armR['blend_full']`, `armR['blend_mlpheavy_anchor']`,
`armR['map512_mlpheavy']` twice, the last inside an f-string with **double** quotes so my own sweep of
the single-quoted form missed it. Every one was a `KeyError` **after** the run finished its GPU work.

I spent real time trying to detect them. **Four static gate checks, all measured against the corpus, all
rejected** (LESSON 82): 218/227 false positives, then 2/82 that skipped the very population at risk, then
35/178, then 43 verdict changes. The check is not the answer.

**The answer is one line of convention.** Bind the reference labels next to the plan that defines them:

```python
PLAN = (('mix25m256', MLPHEAVY, 'blend_mlpheavy'), ...)
REF = 'blend_mlpheavy'        # paired-t and inertness reference
KNEE_LAB = 'blend_768_256'
```

and index with `armR[REF]`, never `armR['blend_mlpheavy']`. A rename then breaks the **definition**, and
an undefined name is exactly what the gate has caught since LESSON 80. **The same mistake changes from
undetectable to detected, with no new checker and no false positives.**

**The general rule.** When a bug class resists detection, check whether it resists because the code is
written in a form the tools cannot see. Moving the information from a string into a binding is usually
cheaper than teaching a checker to read strings — and it is the reason the fifth instance was the last
one. The corollary that cost me a run: when you *do* sweep literals, sweep both quote styles and inside
f-strings.

## LESSON 84 — two bar failures in one tick: one where the code did not implement the docstring, one where the bar was tighter than the reference's published precision

**§1952.** `eff_rank` was registered as *"the largest attention rank still reached by a step buying ≥ 0.010
nats per 100M"* and implemented as *the first* such step. It returned 128 when 128→192 (0.037), 192→256
(0.013) and 256→384 (0.011/0.014) all clear the same bar. **It scored pred_b and pred_c FALSE 0/3; the
corrected helper scores them TRUE 2/3.** The bug inverted the answer, and the gate cannot catch it — the
predicate ran, returned a bool, and meant something else (LESSON 69, third instance). What caught it was
reading the printed per-step rates against the returned value before writing the section. **Print the
intermediate quantity a helper reduces, and check the reduction by hand on one role.**

**§1953.** pred_d required §1932's published 125+ kept-fraction (63.5 / 62.9 / 63.4%) to reproduce within
**0.02pp**. Those figures are published to **0.1pp**, so rounding alone permits **0.05pp** — the bar was
unmeetable by construction. Observed 0.047pp: inside the rounding envelope, outside my bar, **FAIL as
written**. PRE-FLIGHT E says never set a fixed absolute tolerance without scaling by the precision the
data was *computed* in. **This is the other half: scale by the precision the reference was PUBLISHED at.**
A reproduction bar against a 1-dp figure can never be tighter than 0.05pp, and against a 2-dp figure never
tighter than 0.005pp.

**The common shape.** Both are bars that could not do the job they were written for — one because the
code disagreed with the words, one because the arithmetic disagreed with the source. Neither is caught by
"does it run" or "does it return a bool". The check that catches both is **re-deriving what the bar can
possibly measure, before the run, from the numbers it will be compared against.**

## LESSON 85 — I quoted three-role triples after seeing two rows, and filled the third from the second

§1953's write-up stated the converged build "loses the unseen (0-0) bucket pooled on **3/3** — −0.36 /
−0.26 / −0.36pp". The real values are **+0.75 / −0.36 / −0.26pp**: it loses on two roles and **gains**
+0.75pp on the third. Two further triples in the same paragraph (5-24 and 25-124) were also wrong — I had
listed skip11000's and skip1200's values and manufactured a skip7000 entry by repeating skip11000's.

**How.** `tail -22` on the run log showed skip11000 and skip1200; skip7000 had scrolled off. Every
section in this ledger quotes per-role triples, so the shape of the sentence demanded three numbers, and
I produced three. The predicate itself was computed by the script from the full data and was right
(pred_a FALSE 0/3), which is why nothing downstream broke and why I did not notice until §1954 made me
re-derive a claim built on it.

**What caught it.** §1954 asserted "the unseen bucket loses on 6 of 6 role-coverage cells" and I went to
verify that one number before publishing it. It was 5 of 6, and chasing why exposed the fabricated
triple behind it.

**The rule.** A per-role triple is three measurements. **Read all three, from the result JSON, not from
whatever the log tail happened to show.** If the tail shows two, that is not a licence to write three —
`tail` is a display width, not a dataset. And when a section quotes a triple that a *later* section
depends on, re-derive it from the artifact rather than from the prose: the ledger is the record, but the
JSON is the data.

## LESSON 86 — two controls that could not work, and both fixes belong in the library rather than the script

**Polarity, the fourth time.** §1946, §1949, §1951 and now §1955 each inherited the covered-input
inertness control across a fork that changed which arms differed how, and each asserted the wrong
direction. Every time, pred_a/b/c passed 3/3 while pred_d failed on a clause that could not hold. The
polarity is not a judgement call — **§1765/§1936 make it a fact about the plan**: two arms with the same
`table_rank` must be exactly inert at covered inputs, two with different `table_rank` must move them. So
it now comes from `B.inertness_pairs(PLAN)`, which reads the specs and emits both lists. §1955's control
checks **15 same-spec pairs inert and 6 differing-spec pairs not**, and I never chose a direction.

**A cache that skips a computation also skips its side effects.** `Program.routefrac` was populated when
`arm()` built the rows. On a fully warm run `score()` serves every role from cache, `arm()` is never
called, and a control reading `routefrac` **silently read its default** — reporting a routed-fraction
deviation of exactly 0.5000, which looks like a data problem and is an absence-of-data problem. The
watcher went quiet instead of loud, which is PRE-FLIGHT D's second direction. Fixed by computing the
quantity from the signal (`B.route_fraction`) rather than reading it off a side effect.

**The shared rule.** Both bugs are the same shape: a control whose correctness depended on something the
*script author* had to get right — a direction, or a code path having run. Neither survives contact with
a fork or a cache. **Move the invariant into the library where it can be derived, and prefer a control
that computes its own quantity to one that reads a value someone else was supposed to have set.**

## LESSON 87 — I repeated LESSON 85 one section after writing it, so the fix had to stop being a resolution

§1953 quoted a three-role triple with a fabricated entry; LESSON 85 said *"read all three, from the result
JSON, not from whatever the log tail happened to show."* **§1956 then typed
`S1951_CE = (5.94788, 5.93606, 5.94788)` — the first role's value repeated in the third slot** — and its
control failed at 0.0196 against a 0.0005 bar, which reads as a data discrepancy and was an arithmetic
one. Two sections apart, same error, with the lesson already written down between them.

**Why the lesson did not hold.** LESSON 85 asked me to be careful. Care does not survive a fork at
17:16 when the previous section's numbers are three scrolls up and the shape of the constant demands
three floats. **A rule that depends on remembering to apply it is not a fix; it is a resolution.**

**The fix.** `B.ref(results_json, arm, field)` reads a published per-role triple out of the artifact that
produced it. A reference that exists in a JSON is never retyped, and the failure mode disappears rather
than being guarded against. Same shape as LESSON 83 (bind labels as names so a rename breaks a
definition) and LESSON 86 (derive the control's polarity from the plan): **when a mistake recurs, move
the information out of the author's head and into something that can be read.**

## LESSON 88 — the lessons are not the mechanism; ops/test_fast.py is

Eighty-seven lessons are written down and the record shows they are not reliably applied at the moment of
writing a fork: **LESSON 85 was repeated one section after it was written**, and the covered-input control
polarity was inherited backwards **four times** across §1946, §1949, §1951 and §1955. A rule you have to
remember is a resolution. A check that runs in 0.4 seconds is a mechanism.

`ops/test_fast.py` runs in **~0.4s** with no GPU and no model — `BQLIB_NO_MODEL=1` skips the 6.5s model
load that made every pure helper in the library untestable — and it is wired into `ops/enqueue.sh`, so a
broken library or a regressed gate **cannot reach the GPU**. Every check in it is a mistake that already
cost a run:

- `_rk_key` order-independence — two identical builds must not get two cache entries.
- `inertness_pairs` polarity, **and its warning when one side is empty** (§1957 passed a vacuous control).
- `ref()` against a real artifact (LESSONS 85/87, the fabricated triples).
- `paired_t` arithmetic (the instrument §1939 lacked and §1940 used to retract a headline).
- `cost()` against published figures — 224.868M for `nn`, 267.245M for `map512` at 5,419.
- arm-grammar invariants — `nn75` must price identically to `nn75m64` or every cached key lies.
- **six gate fixtures**, each a shape that actually reached the GPU: except-name escape, module-level
  undefined name, a constant assigned twice, too few predicates, and the two that must still pass.

**It caught its first real bug within seconds of being written.** Making the model import optional left
`m` and `DEV` conditionally bound, and the gate flagged every reader of them as undefined — a change I
had just made and would otherwise have shipped. Both directions of the enqueue gate were then tested: a
good script still queues, and deliberately corrupting `_rank_of` blocks every enqueue.

**The rule.** When a lesson recurs, stop writing it down and make it executable. Cheap tests are worth
more than remembered rules because they do not depend on the state of the person writing the fork.

## LESSON 89 — the library removed the boilerplate and left the fork; the fork was the bug

`ops/bqlib.py` cut a run from 267.7s to 4.6s and cut a script from ~430 lines to ~224. It did not cut the
**failures**, because an experiment was still a *copy of the previous experiment*. Eight distinct
fork-residue failures this session: five dropped string keys, four inherited control polarities, three
doubled module assignments, a stale coverage clause, two fabricated reference triples — **none of them a
thinking error, all of them residue from editing ~150 of 224 inherited lines.**

The numbers say the same thing. GPU is now **6%** of wall-clock and authoring **94%**; **90% of the 1,773
scripts ever run were run once**. The tree is an append-only log whose dominant cost is writing the next
entry by editing the last one.

`B.run()` removes the copy. An experiment declares a PLAN, `(key, registered text, fn)` predicates and
its reference anchors. The covered-input control is **derived from the plan**, the coverage assertion is
**generated**, the reference triple is **read from the artifact**, and a predicate's words and code are
**one tuple**. Sixty-six lines instead of 225, and the specific clause that killed the hand-forked twin
(`ncov['c16110']` in a single-coverage run) has nowhere to exist.

**The rule.** When a tool makes the expensive thing cheap, re-measure before assuming you have finished:
the bottleneck moves, and it moved here from compute to authoring in a single day. And when a failure
class keeps recurring under different names, ask what *shape of work* produces it — the answer was "copy
a file and edit it", and no amount of checking a copied file is as good as not copying it.

## LESSON 90 — I proposed archiving 1,601 dead scripts; measuring first killed the proposal

The tree looks like bloat: 247 scripts in `ops/`, 90% of everything ever run was run once. Archiving the
fossils to shrink the live surface is the obvious move. **Measured before moving anything:**

- **238 of 247** scripts are referenced somewhere — the ledger, the registry, or another script. Only
  **9** are provably inert.
- **36** are referenced by another `.py`: real import or path dependencies.
- **237 of 239** result JSONs are referenced by a script or the registry. `B.ref()` anchors read these by
  path, so moving one silently breaks a reproduction control in a script that still looks fine.

**The proposal was wrong.** The ledger and registry refer to scripts *by name* — that is what makes the
record a record — and the artifacts are load-bearing for controls. Moving 9 files is not worth the risk.

**The real bloat problem is not the files that exist; it is that each new experiment adds another
224-line file.** `B.run()` addresses that at the source; the fossils should stay where the record points.

Third time this session that measuring killed a plausible infrastructure plan: caching the 36 tables
(setup was ~4s, saving nothing), four static gate checks for dropped string keys (218/227, then 2/82
skipping the at-risk population, then 35/178, then 43 verdict changes), and now this.
**A tidy-up is a change, and a change needs a measurement, not an intuition.**

## LESSON 91 — significance and magnitude are different questions, and I wrote a bar that conflated them

§1939 was retracted for publishing a headline with no significance test; the fix was to attach a paired t
to every CE claim, and it has held since. §1972 then registered that §1967's tilt would be "not
significant even pooled", with the stated consequence that a failure would mean **"the stopping rule
fired too early"**. It failed — pooled **t = −2.23** — and the stopping rule had **not** fired too early.

**The tilt is worth −0.266 milli-nats: 0.4% of the converged build's 69.238-milli-nat margin over the
deployed design.** It is real *and* negligible. §1967 asked "is this worth buying" and answered on
magnitude, correctly. My predicate asked "is this detectable" and treated that as the same question.

**Why the pooled instrument makes this unavoidable.** On 92,160 positions almost anything real clears 2σ.
A t-value is a statement about whether an effect is nonzero; a build decision needs whether it is worth
its cost. The arc needs both and they do not substitute: **significance to believe an effect, magnitude
to buy it.**

**The rule.** When registering a predicate about whether something *matters*, state the bar in the units
of the decision — nats, parameters, nats per 100M — not in σ. Use σ to decide whether the number is real,
then the units to decide whether it is worth having. And when a predicate's failure is given a
consequence in its own registered text, check that the consequence follows from the bar as written: mine
did not.
