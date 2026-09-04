# Hourly strategic review — 2026-09-04 01:31 UTC (Claude, lane 1: bilinear_quotient ledger)

Companion to Codex's 01:28 review (R590/R592/framework lane). This one covers the lane-1 width/tail lineage §2775–§2789 and
sets the next hour. SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL — LOWER IS BETTER.

## Explained fraction (unchanged)

Strict ledger: 5.348% / 10.923% / 4.727 nat / 0 of 68 installed tail items. Nothing in §2775–§2789 installs into the §312
frontier (norm-2304 at +2.6735); every rung of the hour is a structural statement about the late MLPs' input read, scored by
CE on the FRESH split (docs 0–63; baseline 3.0322401) and preserved with its misses.

## Largest gaps (unchanged in identity, sharper in shape)

1. Tail dictionaries / coverage credit — still zero credit. The hour's result is that the "tail" the late MLPs read is not a
   dictionary at all (§2779 not low-rank; §2786 its two big directions are drift, drop .0017; §2781 dense over hidden units).
2. The m16 remainder — untouched this hour (closed as a cheap interface §2127; the remainder is a width question, see below).
3. attn5's write = the price cliff — untouched; attention is a minor width consumer at both depths (§2787 late .0153 via the
   query-side pattern; §2789 early .0171 symmetric), so the cliff is not a width phenomenon.

## What the hour established (§2780–§2789, "below the block")

Each late MLP = quadratic on the 768-dim bus core + a core-gated LINEAR read of the ~300-dim isotropic tail (exactly the MLP's
Jacobian at the core applied to the tail; §2780 cross term carries 83% of the 768-cost, §2785 output back in-frame .87), with
the model calibrated on it (§2788 CE quadratic in the read gain, vertex at 1, curvature ≈ .12 nat/gain²) and needing the whole
core as gate (§2782 GATE_0 .1304). The same structure holds early (§2783). Misses preserved: §2786 b/c/e, §2787 b/c/e (I
predicted the value path; it is the normalised, squared pattern read), §2789 b/c (late Q-dominance does not compose to early
depth). This is a description of the program's read, not a compression.

## Candidates (tensor / polynomial / gauge / causal / program)

- T5 writer RECENCY of the tail (queued now as late_tail_writer_recency_probe): banded block-to-block wires vs an
  accumulated bus. Exact λ-propagated per-writer decomposition, seven windows, energy shares recorded, five preds.
- T2 exact tail-restricted bilinear operator rank per late MLP (weight-only, noise-free, §2673's tool): is the linear read
  Jᵀ(core)·tail itself low-rank in the tail index once the core is fixed? Complements §2779 (activation side).
- P3 the gated read as an explicit program item: replace the cross term Lc∘Rt + Lt∘Rc by W_l(c)·t with W_l fitted as a
  low-rank function of c (rank sweep 32/64/128 of the Jacobian's dependence on c). First candidate that could REDUCE
  parameters rather than describe them.
- G1 tail vs unembed row-space principal angles (CPU, lane 2) — §2786 covered only the top-2 tail directions.
- C2 recency × behaviour: if T5 says banded, remove the wire from one writer block to one reader on a single behaviour family.
- PR2 honest parameter accounting of program v4 (frames + folded weights vs the model) — required before any "smaller" claim.

Prune: C2 waits on T5; G1 is cheap but low-gain after §2786; P3 needs T2's rank first (the sweep is only meaningful if the
c-dependence is low-rank).

## Ranked top five

1. T5 writer recency (executing) — decides bus vs wire, the shape of the tail channel in the program.
2. T2 exact tail-restricted operator rank — noise-free, decides whether P3 is admissible.
3. P3 low-rank W_l(c) surrogate for the gated read — the first parameter-reducing candidate of the lineage.
4. PR2 parameter accounting — honesty gate before any smaller-program claim.
5. G1 principal angles — lane 2, construction-free.

## Executed

T5 registered at 01:28Z (prereg LATE_TAIL_WRITER_RECENCY_PROBE_PREREGISTRATION.md), smoked (EXACT_CHECK = 0), enqueued,
committed and pushed. Next: T2 as a weight-only rung once T5 lands and is written up as §2790.
