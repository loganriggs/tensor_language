# Standing benchmark backlog (never let a wake idle while any of these is open)

Honest frontier: 34/36 at +2.93 fresh (75% of model work). Open rungs, by size:
1. FRONT TRANSFER TAX (+1.98 of the +2.93 on fresh): replace empirical tables
   with FOLD tables (weights-derived, transfer-free per §283); refit absorbers
   small. -> fold_front.py
2. MIDDLE-ATTENTION BAND (~0.9 in-context): head-level hybrid (motif heads
   swapped, diffuse real) -> head_hybrid.py [QUEUED]; then absorbers on the
   attention dictionaries (they have none).
3. a8 / COUNTING: symbolic count features from raw tokens (deploy-legal)
   -> a8_symbolic.py [QUEUED]
4. DEPLOY GAP (~0.09): two-probe labeling (a10-input probe acc 0.73 for the
   late rungs) -> two_probe_deploy.py
5. INDUCTION heads: closed as irreducible-linear, but a *bilinear* stand-in
   (learned low-rank pattern q-k factor + value read) was never tried at the
   head level.
6. Fresh-window certification for every new winner (ledger 22), figure + report
   updates at each frontier move.

Rule (from 2026-08-18 stall): "science arc closed" NEVER implies "benchmark
saturated". A wake with an empty queue must pull from this file first.
