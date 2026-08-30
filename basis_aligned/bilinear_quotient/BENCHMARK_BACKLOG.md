# Standing benchmark backlog (never let a wake idle while any of these is open)

Honest frontier: 34/36 at +2.93 fresh (75% of model work). Open rungs, by size:
STATUS AUDITED 2026-08-30 against runlogs/_completed.txt and the ledger, because
the [QUEUED] markers below were stale and a wake had begun re-deriving them.
Rungs 1-3 are DONE; rung 4 is the first genuinely open one.

1. FRONT TRANSFER TAX (+1.98 of the +2.93 on fresh): replace empirical tables
   with FOLD tables (weights-derived, transfer-free per §283); refit absorbers
   small. -> fold_front.py
   **DONE** -- 2 completions (21:35 exit=1, 21:38 exit=0; the nonzero exit was
   not a failed experiment, per PRE-FLIGHT F). Written up as ledger **§305**,
   "Fold tables strictly dominate: the transfer tax was avoidable all along":
   both bars HELD with margin, window C +1.390 vs +1.481 and fresh +1.602 vs
   +1.926 -- a third of a nat of transfer tax removed.
2. MIDDLE-ATTENTION BAND (~0.9 in-context): head-level hybrid (motif heads
   swapped, diffuse real) -> head_hybrid.py; then absorbers on the attention
   dictionaries (they have none).
   **DONE** -- ledger **§303** (both bars HELD, +2.3644 on window C, 0.27
   BETTER than the all-dictionary band) and **§304** for head_hybrid_fresh
   (certified on never-seen text). The absorbers-on-attention-dictionaries
   half is NOT done and is the live remainder of this rung.
3. a8 / COUNTING: symbolic count features from raw tokens (deploy-legal)
   -> a8_symbolic.py
   **DONE, and the result was NEGATIVE** -- ledger **§304**: "the
   counting-feature rescue is REFUTED with unusual" force. Closed, not open.
4. DEPLOY GAP (~0.09): two-probe labeling (a10-input probe acc 0.73 for the
   late rungs) -> two_probe_deploy.py
   **OPEN, but far more advanced than this line suggests -- RESEARCHED
   2026-08-30, read this before building anything.** `two_probe_deploy.py`
   does not exist and never ran, but the gating ladder it belongs to has four
   measured rungs and one failed branch:
     - **§337** 14 input-only surface programs: efficiency **1.5x** random,
       both bars FAILED. (`experiments/gating/deploy_gated.py`)
     - **§341** 57 surface programs: gain 12x larger (+0.171) but efficiency
       still **1.54x**. Verdict: "the deploy gap is a property of the
       DESCRIPTION LANGUAGE, not of program count."
       (`experiments/gating/deploy_gated2.py`)
     - **§342** `probe_gate.py`, linear ridge on the residual stream after
       block 2, fitted to fit-window oracle labels: **3.8x random (bar 2.5x
       HELD)** -- the first deploy-legal gate above the surface ceiling. AUC
       0.621 (bar 0.75 FAILED), 42% of oracle gain (bar 50% FAILED, close).
     - **§347** `probe_gate2.py`, quadratic features in mlp3's read directions
       + per-mode regression: **strictly WORSE than v1** -- AUC 0.551,
       efficiency **-0.625** (worse than random), all three bars FAILED.
     - oracle causal labels: **9.4x**, not deploy-legal.
   **The live branch is named by §347's own closing line:** "the AUC ceiling
   (~0.62) needs information not linearly-or-quadratically present at block 2
   -- **later read points** or context aggregation, not fancier features."
   §342 had set the same fork ("deeper reads OR nonlinear features"), and
   **nonlinear was tried and failed; deeper reads were never tried.** An
   a10-input probe IS the deeper read, so rung 4 is that fork's untaken half.
   **Design constraint to settle first:** §342's block-2 read is deploy-legal
   "because every frontier config keeps the cheap lexical rungs real". Reading
   at a10's input requires blocks 0-9 to be computable at deploy; whether the
   frontier assembly supplies them real or via stand-ins decides whether an
   a10 probe is §105-legal at all, and that must be checked before fitting.
   **Related negative worth knowing:** §1365 (`exclaim_probe_gate2.py`) tried
   TWO capture sites (L3 + L8 concatenated) on a different capability and got
   AUC 0.611 vs 0.618 -- no gain -- because "a kit-stream probe cannot see
   what the kit removed". If the a10 probe reads a stream whose relevant
   values the assembly ablates, it inherits that ceiling by construction.
5. INDUCTION heads: closed as irreducible-linear, but a *bilinear* stand-in
   (learned low-rank pattern q-k factor + value read) was never tried at the
   head level.
6. Fresh-window certification for every new winner (ledger 22), figure + report
   updates at each frontier move.

Rule (from 2026-08-18 stall): "science arc closed" NEVER implies "benchmark
saturated". A wake with an empty queue must pull from this file first.

7. MODE-CONDITIONED STAND-IN SELECTION (from §§315-322): damage modes are
   component-axis objects. Design: per-position gating -- at positions of
   mode M (causally labeled), keep M's implicated components real (or fit
   M-sliced absorbers); elsewhere use the cheap stand-ins. Register: the
   gated assembly beats the uniform one at matched parameter budget;
   control = random position gating at matched fraction. Requires a
   deployable mode-labeler (probe on the stream -> mode score) -- fit and
   validate that first (deploy gap rules apply).

8. PATTERN-SIDE MECHANISM RUNG (from §332): attention-probed leaves need
   mechanism conditions built from motif patterns composed with value
   reads (e.g. "prev-motif head at L, values carrying X -> fires when
   previous token writes X"). Design before running; this is the ladder
   rung for the census majority.
