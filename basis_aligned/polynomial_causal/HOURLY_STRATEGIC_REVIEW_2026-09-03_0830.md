# Hourly strategic review — 2026-09-03 08:30 UTC (Claude) — DECISION CHECKPOINT

Sign convention §2135: frontier L2 = CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER; frontier §312 norm-2304 at
+2.6735. Role split: Codex leads direction + owns GPU (rung522 VALIDATION census live; R523 optimizer repair
building); Claude red-teams + CPU probes + ops.

## The checkpoint this review exists to surface

Explained fraction unchanged: 5.348% / 10.923% / 4.727 nat / 0 of 68. Rungs 506-522 (~16 rungs, the entire
equality-score / attention8 circuit-resolution campaign) have produced ZERO explained-fraction gain, and the
MLP10 estimation chapter (§2657-§2668, closed this session) now proves WHY, rigorously:
- Per-node circuit fingerprints are cross-half noise (rho 0.016, §2657); every per-unit grouping test is
  attenuation-capped regardless of truth.
- The reliable shared structure that pooling recovers is SMALL: prequential MDL says the effect matrix is not
  low-rank, held-out coverage is 12% (not the soft 76%), and it saves ~0 bits (§2668).
- The whole-attention8 response is reliable but BROAD/non-selective (§2665); rung522's attempt to isolate a
  selective sub-projector just failed as an INVALID OPTIMIZER (loss to 270M, Codex's diagnosis — a genuine
  instrument failure, not a circuit null), and is being repaired as R523.

This is not a run of bad luck; it is a measured wall. The per-unit circuit-resolution line is data/power-bound.

## The fork (a genuine invest-vs-pivot decision, surfaced for Logan)

There are exactly two ways forward, and they cost very differently:
- **INVEST:** the raise-N re-measure I preregistered (MLP10_RAISE_N_COVERAGE_REMEASURE_RUNG, ~122k forwards)
  is the decisive test of whether the 12% ceiling is N-LIMITED (fixable with ~26-62x documents, §2659) or
  FUNDAMENTAL. If N-limited, the whole per-unit line reopens with real data. If fundamental, close it.
- **PIVOT:** accept the ceiling and change object — either a coarser, higher-signal downstream readout than
  per-source circuit effects, or a different module/structure, or return to the deployed-compression frontier
  (§312, which DID achieve real byte savings) with the interpretability lens.
Codex has (reasonably) declined the raise-N experiment while repairing rung522's optimizer (R523). The
invest-vs-pivot call is a human-judgment strategic decision worth Logan's input; both agents can keep executing
in the meantime.

## Largest gaps (unchanged; CPU-status noted)
1. Tail / COVERAGE CREDIT — ADVANCED: §2668's MDL frame gives the principled bits-saved metric; recent circuit
   results earn ~0 bits by it. Framework now exists.
2. m16 remainder — CPU-blocked (no committed bundle).
3. attn5 write price cliff — CPU-blocked; off current steering.

## Ranked top five
1. **Raise-N re-measure (decisive N-limited-vs-fundamental test) — PREREGISTERED (0730), awaiting the
   invest-vs-pivot decision.** GPU/Codex lane.
2. **R523 optimizer repair** — Codex's active lane (correct next step for rung522).
3. **rung522 post-hoc red-team** — execute when a HEALTHY rung522 lands (against §2665/§2668 backdrop).
4. **Pivot to a higher-signal object** — propose; needs a GPU re-measure with a coarser readout.
5. **Coverage-credit MDL accounting** — CPU, deferred (bookkeeping; §2668 already gives the recent-arc answer).

## Executed
No new CPU probe enqueued (honest determination, consistent across recent wakes): the runner is busy on rung522's
VALIDATION census; the MLP10 CPU line is closed and capstoned; cross-module is data-blocked; frontier/weight-space
are off Logan's steering or closed; and the rung522 spike diagnosis is Codex's active R523 lane. Forcing a
redundant probe after this session's over-produce->correct pattern would be negative value. The substantive act
this review is the DECISION CHECKPOINT above, surfaced to Logan and Codex on the board. Waiter armed for the
terminal rung522 receipt and R523.
