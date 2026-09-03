# Hourly strategic review — 2026-09-03 16:38 UTC (Claude) — the price map says: fine-grain the LATE blocks, treat the EARLY MLPs as dense

Sign convention §2135: frontier L2 = CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER; frontier §312 norm-2304 at +2.6735.
Role split: Codex leads direction (pending-opener R549-R551, induction factorial R552-R554); Claude red-teams + CPU probes + ops.
Codex's own 16:30 review (HOURLY_STRATEGIC_REVIEW_2026-09-03_1630.md) stands as the lane's direction; this file is the
red-team/CPU-lane complement and does not restate its seven targets.

## Explained fraction (strict ledger, unchanged)
5.348% / 10.923% / 4.727 nat / 0 of 68. Nothing landed this hour installs into the §312 frontier or the strict ledger.

## What landed since 15:30 (my lane)
- §2694 (15:56): MLP16/17 write PCA rank-8 costs .036 / .083; both .172 (super-additive); entropy-dial falsified.
- §2695 (16:25): ledger coverage of Codex R537-R548 (site L13H8 complete head; R540/R544 strong nulls).
- §2696 (16:35): the 36-site k=32 truncation map. Variance rank does NOT order causal price (ρ .23; MLP-internal −.43).
  Depth does (ρ −.81 MLP, −.88 attn). Price concentrates in mlp1 .883 / mlp2 .220 / mlp0 .165 / mlp3 .130 (59% of
  2.371). The eight highest-rank writes (mlp7-14, eff rank 559-679) each cost < .045.
- Math review 16:30 (MATHEMATICAL_REVIEW_2026-09-03_1630.md): RMSNorm scale gauge + Fisher pull-back certificate;
  Fisher-certificate probe registered 16:31 (smoke in progress, enqueue next).

## Largest gaps (restated, with what §2696 changes)
1. Tail dictionaries / coverage credit — attention-side, §2668 MDL frame. §2696 adds: attention writes are all cheap at
   rank 32 (sum .429 over 18 sites), attn0/4/5 the only ones ≥ .045 — the tail credit is a bookkeeping gap, not a
   causal-density gap.
2. m16 remainder — §2694/§2696: mlp16's write is functionally ~rank-32 (.028) but rank-8 leaves .036 and the fat tail
   is what §2127 priced; interaction-term view (quadratic-form probe, running) is the live test.
3. attn5's write = the frontier price cliff — §2696 puts attn5 at .047 single-site at rank 32 (eff rank 110), i.e. the
   cliff is not a single-site write-rank cliff; it is a JOINT-installation/tail effect with the early MLPs, whose
   single-site prices dominate (mlp1 .883). This reframes the cliff as an early-block problem.

## Brainstorm → prune → rank (top five)
1. Fisher-certificate probe (registered 16:31) — EXECUTING: it turns the §2696 "downstream reads a low-dim slice"
   fact into a local, second-order, per-direction price certificate (pred_c ratio in [.5, 2]) and a Fisher-metric basis
   (pred_d ≤ .05 for MLP17 k=8, vs PCA's .083). If pred_c holds the map becomes computable WITHOUT forward passes per
   candidate edit — the editing tool Logan asked for. Closure note: this is a cfgE/late-MLP tool, NOT a frontier
   half-price retry (§2118/§2125).
2. Early-MLP dense-write diagnosis (NEW, CPU, cheap): for mlp1 (.883 at k=32) measure the rank-k ladder k ∈ {64, 128,
   256, 512, 1152} on the same eval half, AND the same ladder with the write replaced by its per-token mean (kills the
   context branch only) — decides whether mlp1's causal density is in the token-only branch (compilable by a lookup
   table, §2673 frame) or in the token-context operator (exact-high-rank, §2675). Preregister next wake; est. 15 min CPU.
3. Red-team R549 atlas on landing (Codex's; landed 16:35, 13 s, "next_step: retain endpoint plus invariance objective
   without claiming a downstream reader") — read the results JSON, check the R551 readout-span rejection rule was
   applied as frozen, write the ledger § if Codex does not.
4. Joint-installation map for the cheap late sites (mlp7-17 + all attention at k=32, one bundle): §2694 says
   super-additivity is real; a single joint number bounds how much of the late network is compilable into 32-d writes.
   CPU ~2 min per forward set; preregister after (1) lands to avoid stacking three CPU probes on the lane (ops finding).
5. Lane-2 CPU-only runner (ops proposal on the board): my 30-min CPU probes held Codex's 13-s GPU rung R549 for 32 min.

Pruned: any further variance-rank / PCA sweep across sites (§2696 closes the "rank predicts price" question: FALSE);
metric-constructed spans on the frontier (§2118, CLOSED); conditioning on cfgE (§2132, zero).

## Executed this wake
- §2696 written and scored as registered (a TRUE, b/c/d FALSE, d's null held); §2694 baseline-label correction appended.
- Fisher-certificate probe: smoke running (6 threads, runner CPU-bound on the quadratic-form probe; accepted because the
  queue depth was 0 pending) → enqueue on pass; queue then ≥ 2 with the quadratic-form probe in flight.
- Ops: lane-2 finding + proposal logged (ops/EFFICIENCY_LOG.md 16:37 row).
