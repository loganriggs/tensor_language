# Repository agent guidance

Read `AGENT_BOARD.md` before claiming work. It is append-only; coordinate there and
do not collide with shared GPU queues.

The program-level objective is to reverse engineer the entire bilin18 model as a
small, executable, causally faithful tensor program. Local experiments are valuable
only insofar as they reduce uncertainty or description length in that whole-model
program. Read `WHOLE_MODEL_STRATEGY.md` before choosing a new research thread.

## Hourly strategic review

At the start of a work session, and at least once per hour during continuous work,
check whether `.strategy-review-due` exists. If it does—or if an hour has elapsed
since the last review—pause local work and answer the checklist in
`WHOLE_MODEL_STRATEGY.md` before selecting the next action. Record a concise review
at the bottom of `STRATEGIC_REVIEW_LOG.md`, then remove the untracked due marker.

The review must be substantive. In particular, ask whether the current task:

- advances whole-model coverage or merely deepens one already-understood example;
- creates a reusable representation/compiler primitive or a one-off measurement;
- tests end-to-end composed fidelity, not only isolated causal damage;
- uses the tensor-network, arithmetic-circuit, system-identification, or MDL
  literature before inventing an ad hoc method;
- targets the current highest-value uncertainty in the coverage ledger.

Never let the reminder itself launch an experiment, Codex process, network action,
or GPU job. It only forces strategic reconsideration; normal authorization and GPU
coordination rules still apply.

## Published explanation handoff

The reader-facing MLP0--4 and attention explanations are published on branch
`codex-explanations-handoff`. This branch is based directly on the current
`origin/main` and contains one additive handoff commit: `explanations/` plus only the
evidence and CPU implementation files linked from those chapters. It intentionally
excludes the large experimental history of `codex-local-simplicity-audit`, so other
agents can fetch, review, or cherry-pick it without interfering with their branches.
