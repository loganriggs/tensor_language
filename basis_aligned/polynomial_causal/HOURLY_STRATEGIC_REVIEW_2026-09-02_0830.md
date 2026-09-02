# Hourly strategic review — 2026-09-02 08:30 UTC

## Hour's landings (all ledgered through §2599)
- Projector-form factorial: pred_a FALSE (.11–.20 nat singleton bridge),
  preserved as written; science reported-unclaimable.
- Two diagnostic probes: L1–L4 exact, then the decisive one — **474's own
  collect_window, zero copied code, fresh-vs-bundle .084 nat** (all seven
  subsets .038–.084) ⇒ TEMPORAL REPRODUCIBILITY BREACH declared (board
  08:56, §2599).
- Codex: 480 registered/implemented, one self-repaired crash (08:24→08:25),
  retry running since 08:26 and holding the lane.

## NEW FORENSIC FINDING (this review's execution, CPU-only)
The canary time-series from git history is **bit-identical to 10+ decimals
across every run from 01:59 through 08:28** (score_rank 4.0746173859,
l1_cost 0.2750666142, ratio_5_6 0.2525360633, ratio_14_15 0.0431063355 —
fourteen consecutive canary receipts, spanning the breach window). The
canary runs real model forwards + eigendecompositions through the same
runner. Therefore:
- The GPU/toolchain did NOT globally flip. Determinism per se is intact.
- The breach is SPECIFIC to the 472/474 machinery's computation path.
- And it is NOT explained by changed inputs: `git diff 05:57..HEAD` shows
  every changed file is a post-474 rung output or my instruments — nothing
  the ≤473 chain reads; every hash-checked load passed in the 08:02+ runs;
  the only untracked artifact is the gitignored 479 bundle.

## Surviving hypotheses (two, now testable)
- H-A **per-process kernel selection**: cuBLAS/cuDNN algorithm choice for
  the einsum/hook-laden 474 path varies with process memory state; the
  canary's shapes happen to be stable. Prediction: two fresh probe runs
  DISAGREE with each other at the same order (.0x nat).
- H-B **stable post-05:57 state shift** in something the 474 path touches
  but the canary doesn't (e.g., a lazily-written cache outside git).
  Prediction: fresh probe runs agree with each other bit-exactly and all
  differ from the bundle identically.
Decider ENQUEUED: the replication probe re-run (behind 480); run1's
full-precision per-subset maxima are recorded here for comparison
(m8 0.08405804634094238, m9 0.06967806816101074, m12 0.03833127021789551,
m8+m9 0.05142807960510254, m8+m12 0.0832054615020752,
m9+m12 0.04619002342224121, union 0.06456136703491211).

## Program state
Strict ledger explained fraction unchanged (5.348% / 10.923% / 4.727 nat /
0 of 68). The equality-MLP arc is closed at every grain (§2595–§2598); the
active decomposition line is 480 (attention0 continuous block, running).

## Ranked top five
1. **480** (Codex, running): score as written on landing; verify no clause
   trusts a stored cross-session number below .1 nat (§2599 caveat).
2. **Reproducibility decider** (mine, enqueued): H-A vs H-B from probe
   run2 vs run1 full-precision comparison; canary fingerprint v2 baseline
   lands next idle cycle and time-locates any future flip.
3. **Projector b-variant** (mine, blocked on 2): re-register with IN-RUN
   subtractive baseline (L4-style, proven ≤2e-6) instead of the
   cross-session bundle; its unclaimed code cosine .9389 vs replacement
   +.9793 makes pred_b live again under a sound instrument.
4. **Support-stability audit** for any 480 survivor (standing).
5. **Natural-register equality mechanics** (parked).

Pruned: everything rank/CP (00:47 direction), info bottleneck, archetypal,
Hankel — per math review 0713; no change.
