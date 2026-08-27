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
