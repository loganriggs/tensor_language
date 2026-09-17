# Claude hourly step-back review — protocol

Written 2026-09-17 at Logan's request: "an hourly job to take a higher level view to avoid
getting stuck in a local direction that doesn't help, as well as an efficiency measure of what
you've done and how you can improve the process (organizing, refactoring, caching)."

Each review is a file `REVIEW_<UTC yyyy-mm-dd_HHMM>.md` in this directory, ≤ 60 lines. It is
written from receipts on disk, not from memory of intent. A review with no executed
consequence is allowed only if it states why nothing should change.

## Part A — higher-level view (anti-local-minimum)

1. **Goal restatement in one line**, from `basis_aligned/better_circuits.md` §1: a circuit is
   done only when it is *simple, predicts OOD, extracted, selective, composes* — each against a
   preregistered gate and a null. Partial paths, folds, fits and single edits are not circuits.
2. **What changed since the last review** (receipts, board entries, commits) — list the files.
3. **Definition-of-done scorecard** for the circuit(s) I am driving: five properties × {held,
   failed, untested}. Did the hour move any cell from untested/failed to held? If not, why?
4. **Local-direction check.** Answer explicitly: (a) is the current thread a new behaviour or
   dataset (forbidden by better_circuits §7 until the current target is done)? (b) am I
   improving replay error on opened rows instead of moving a property? (c) am I building a
   fitted proxy and calling it a component? (d) is the next step the highest information per
   GPU-second toward an unheld property, or just the easiest? (e) does it collide with
   Codex's board claims?
5. **Kill criteria.** For the current thread, state what result would make me stop it, and
   whether that result has already occurred.
6. **Decision**: continue / redirect / stop, with the one next action.

## Part B — efficiency measure

1. **Time accounting** for the hour, from timestamps in receipts, runner logs and board
   entries: minutes on (i) reading/orientation, (ii) writing scripts/preregs, (iii) waiting on
   runs, (iv) analysis/interpretation, (v) reviews/ceremony. Science = ii+iv; if ceremony or
   orientation is the largest bucket, name the fix.
2. **GPU accounting**: forwards and GPU-seconds spent (from `runlogs/` and result timings),
   and how many produced a decision-changing number.
3. **Repeated work**: any code copied rather than imported, any recomputation that a cache
   would remove (native forwards, donor caches, tokenized rows), any file I re-read that a
   note would have replaced.
4. **One bounded process improvement**, executed or scheduled: a refactor, a cache, an index,
   a note in this directory, or a shared helper. State it in one line with the file path.

## Rules

- Never backfill hours I was not working. Never fabricate timings.
- Prefer citing a result JSON path over restating its numbers.
- Preserve nulls, failures and retractions; do not soften them in the summary.
