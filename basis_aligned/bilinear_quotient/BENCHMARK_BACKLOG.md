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
   **RUN 2026-08-30 as `ops/probe_gate3.py`; written up as ledger §2079.**
   §342 set the fork as "deeper reads OR nonlinear features"; §347 took the
   nonlinear half and came out strictly worse, closing with "the AUC ceiling
   (~0.62) needs LATER READ POINTS". That half had never been run. v1's read
   depth was a single constant, so the derivation changed only the depths.
     blk2 (v1 replicate)  AUC 0.6210  eff 3.867x  frac_of_oracle 0.420
     blk5                 AUC 0.6384  eff 5.488x  frac_of_oracle 0.596
     blk9 (= a10 input)   AUC 0.6545  eff 4.574x  frac_of_oracle 0.497
     blk2+9 (two-site)    AUC 0.6382  eff 6.113x  frac_of_oracle 0.664
   **Depth breaks the ceiling** (0.6545 > 0.621) and **the deploy-legal ladder
   is now 1.5x -> 3.87x -> 6.11x against a 9.4x non-deploy-legal oracle.**
   §342's original "recover >= 50% of oracle gain" bar, which v1 FAILED at
   41.9%, is HELD by blk5 (59.6%) and blk2+9 (66.4%).
   **Two sites FAILED on AUC** (0.6382 < 0.6545), reproducing §1365's two-site
   negative in a second program -- **but won on every gating measure.** AUC
   scores the whole ranking; a gate at 17.25% uses only the top of it, so
   **§342's AUC >= 0.75 bar was the wrong bar** and gain-at-fraction is the
   deployable quantity.
   **STILL OPEN on this rung, and it is a control not a new idea:** [2;9] is
   2304-dim against the single sites' 1152, so depth-pair versus width is
   unseparated. Queued as `ops/probe_gate4.py` (all three 2304-dim pairs).
   **Also owed: fresh-window certification (rung 6) before 6.11x is quotable
   as a frontier number.**

5. INDUCTION heads: closed as irreducible-linear, but a *bilinear* stand-in
   (learned low-rank pattern q-k factor + value read) was never tried at the
   head level.
6. Fresh-window certification for every new winner (ledger 22), figure + report
   updates at each frontier move.
   **2026-08-30, §2081: this rung is not merely open, it is OWED RETROSPECTIVELY.**
   Certifying the rung-4 gate on a second document-disjoint fresh window failed
   all three predicates -- and the failure was not confined to the new work:
   §342's published 3.8x configuration scores **-0.093x (worse than random)**
   and the 9.4x oracle reads **2.647x**, because the random-gating denominator
   is **2.595x larger** on the second window. **Every "Nx random" figure in the
   gating arc is a ratio whose denominator was never certified.** Sections are
   scoped, not deleted -- the arithmetic was right on each section's own window.
   **Before any gating number is quoted again it needs a second-window check,
   and the headline should be gain-at-fraction in nats** (blk2+9: +0.0517 FR,
   +0.0331 FR2) which does not divide by a moving baseline. LESSON 113.

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
