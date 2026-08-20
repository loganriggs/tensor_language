# SWARM RUNBOOK — multi-day operation (driver: Opus 5; workers: Sonnet)

Written 2026-08-20 for the model swap. The driver runs the wake loop and
orchestrates waves; workers are FRESH stateless Sonnet agents (Agent tool,
model "sonnet"), one circuit each, per [[fresh-agents-per-batch]]. The
goal is SLOW AND STEADY: a few certified, reviewed records per wave, every
wave pushed, no wave depending on the driver's memory of the last one.

## 0. New-session bootstrap (do once per driver session)
1. `supervisorctl status bqrunner` — RESTART if not RUNNING. bqrunner
   survives session death; the cron does NOT.
2. Recreate the wake cron (session-only!): CronCreate, schedule
   `7,22,37,52 * * * *`, prompt = the standing wake message (see git log
   or the previous cron text in MEMORY). Without it, writeups stall but
   bqrunner keeps the GPU busy.
3. `git pull` then `git log --oneline -10` + tail of BILIN18_CONNECTION.md
   to load current state. NEVER work from memory of an old session.
4. Artifact: the published report is
   https://claude.ai/code/artifact/60bcb3fd-b383-4ceb-b242-9a6a4f24addb
   — pass this as `url` when republishing from a new session.

## 1. The wave loop (repeat ~every 1-2 hours while credits allow)
1. CONSOLIDATE: check runlogs/_completed.txt; write up finished queue
   runs (registered bars scored honestly); commit AND push (box is not
   volume-backed).
2. AUTHOR WAVE: pick up to 4 unclaimed tags from swarm_shortlist.json
   (skip depth-0; skip tags with a circuits/ file; prefer packed tags).
   Launch up to 4 Sonnet agents IN ONE MESSAGE with the author prompt
   (§3). Cap 4: the GPU is 32GB and each python process loads its own
   model copy (~2GB + activations); >6 concurrent risks OOM.
3. REVIEW WAVE: for every merged record with no review yet, launch a
   Sonnet reviewer (§4) — different agent than the author, always.
4. CONSOLIDATE AGAIN: on notifications, verify each record file exists
   and parses, spot-check one number, commit all records + reviews, push.
   The driver — not workers — is the only committer.
5. QUEUE: keep >=1 registered experiment in queue.txt (absolute paths).
   Mechanism-first per MEMORY: named variables/writers/couriers, not
   k-laws or surface programs.

## 2. Standards that do not relax when models get smaller
- Predictions registered in docstrings BEFORE running; controls and
  nulls always; bars scored as written, misses recorded as FAILED even
  by 0.001 (see 391, 405, 406 for the house style).
- An arm that cannot fail is not evidence (the 401 tautology). A null
  that passes disqualifies the headline even when bars held (412).
- Corrections stated plainly, propagated to the published report.
- Workers NEVER git commit/push (concurrent-writer sweep hazard).
- INFRA FREEZE during waves: no BEHAVIOR-CHANGING edit to
  census_lib/SOP while workers are in flight -- a mid-wave edit
  changed leaf_program's numbers between one agent's two calls and
  read as nondeterminism. Purely ADDITIVE helpers (new functions,
  new scripts) are allowed mid-wave; anything that alters an
  existing function's output waits for the gap between waves.

## 3. Author prompt template (fill TAG; keep lean)
> You are a swarm circuit agent for the bilin18 interpretability
> program. Work directory: /workspace/tensor_language/basis_aligned/
> bilinear_quotient. Your assignment: produce one merged circuit record
> for leaf tag "TAG" on the DIVERSE census tree. Read CIRCUIT_SOP.md
> (v3 -- the MECHANISM step 3M is the deliverable; behavioral
> stories need cl.story_test base-rate clearance) and follow it
> exactly, in order. Setup lines are mandatory
> (`source /venv/main/bin/activate`; `import census_lib as cl;
> cl.use_state(cl.PT+'census_state_diverse.pt')`). If your tag has a
> pack, verify concentration reproduces (only concentration — pack
> counts are a 60-row subsample), then continue. On CUDA OOM wait 60s,
> retry once, then report and stop. NEVER PARK ON A BACKGROUND JOB
> (wave-4 lesson: two agents stalled waiting on a slow step while
> four siblings shared the GPU) -- run steps inline, and if one is
> slow, record "not computed under swarm load; demoted step, not
> blocking" and finish the record. DO NOT git commit or push.
> Report: verification numbers, story, red-team hit count, record
> path, and any SOP friction.

## 4. Reviewer-two (adversarial; every record gets one)
Reviewer = fresh Sonnet agent that did NOT author the record. Prompt:
> You are REVIEWER TWO for the bilin18 circuit program — adversarial by
> assignment: your job is to REFUTE the record, not to polish it. Work
> directory: /workspace/tensor_language/basis_aligned/bilinear_quotient.
> Target record: circuits/FILE.json. Do, in order: (1) recompute the
> gate: rerun cl.leaf_ablate + cl.sign_stats (setup: activate /venv/main;
> `import census_lib as cl; cl.use_state(cl.PT+'census_state_diverse
> .pt')`) and check concentration is within 20% of the recorded value;
> (2) test the story: draw 5 FRESH member examples with cl.examples(tag,
> d, ntop=0, nrand=5, seed=11) — a seed the author never used — and
> score the story's help/hurt prediction against each example's dCE
> sign; (3) hunt gerrymander: does the story predict anything, or would
> "helps members" fit any leaf? State the strongest objection you found
> even if the record survives. Verdict rules (v2, from the 414 dry
> run): REFUTE if the gate fails to reproduce. Otherwise start from the
> fresh-example score, then apply two demotions: (i) if the story's
> SPECIFIC claim (the class it names) was never exercised by your draw
> — all hits came from the catch-all branch — the verdict is at most
> WEAKEN unless you draw 5 more examples FROM the named class (filter
> members by the claim) and they score >=4/5; (ii) if the record's own
> program_bacc FAILED its bar, CONFIRM likewise requires the
> class-targeted draw at >=4/5. CONFIRM only when the gate reproduces
> AND the specific claim survived a draw that could have killed it.
> Append your review via cl.write_circuit(tag, {'certification':
> [{'test':'reviewer-two','source':'sonnet-reviewer','date':'<UTC
> date>','verdict':..., 'concentration_recheck':...,
> 'story_fresh_hits':'k/5','objection':...}]}) — the dedup key is
> (test, source, date); OMITTING THEM makes a second reviewer's entry
> silently drop as a duplicate. Note: cl.examples(tag, d, ...) needs
> d=cl.leaf_ablate(tag) passed in, else dce fields are silently
> omitted. write_circuit APPENDS certification entries, never
> overwrites. DO NOT edit any other field or any other file. DO NOT
> commit. Report the verdict and the objection.
Driver applies verdicts: REFUTE -> record's story stripped to gate-only,
tag returned to the pool with a note; WEAKEN -> story flagged; CONFIRM ->
counts toward the certified tally.

## 5. Health checks (each wave)
- `supervisorctl status bqrunner` RUNNING; `df -h /workspace` (>10G
  free); `nvidia-smi` (no orphaned full-memory processes).
- runlogs/bilin18_canary2.log still cycling (model + data sanity).
- If a worker is silent >30 min: it died mid-GPU-wait; relaunch its tag
  once, else skip the tag with a note.
- If a worker REPORTS but says it is waiting on a background job, it
  has parked: SendMessage it to finish without blocking (see the
  author template's never-park rule). Two wave-4 agents did this.
- Heavy QUEUE scripts must tolerate a swarm: wrap per-item work in
  try/except, save results incrementally, and support resume --
  gate_specificity was killed twice by GPU pressure before it was
  made resumable.

## 6. Known traps (union of every incident so far)
- NAMESPACE COLLISION: circuits/ holds ~50 OLD-TREE (212-row)
  records whose tags overlap the diverse tree's (e.g. r.2.0,
  r.1.0). A diverse-tree write_circuit on such a tag would MERGE
  INTO the old record and mix trees. "Skip tags with an existing
  circuits/ file" is the operative rule; check doc['tree']
  ['instance'] before ever merging into an existing file.
- queue.txt: ABSOLUTE paths only.
- cl functions without use_state() silently run the OLD 212-row tree.
- Pack examples are non-canonical; regenerate with cl.examples.
- Depth-0 tags: concentration ill-defined; skip.
- Raw damage-profile cosine is NOT the identity gate (404).
- Row parity is NOT doc-disjoint on the diverse corpus; use docid parity.
- Artifact publishes from a new session need the url= parameter.
- Report body edits: the artifact conflict-409 needs re-read/merge, not
  force, unless the published copy is verified identical to git base.
