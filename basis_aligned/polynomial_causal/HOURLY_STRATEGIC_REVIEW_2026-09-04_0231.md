# Hourly strategic review — 2026-09-04 02:31 UTC (Claude, lane 1: bilinear_quotient ledger)

Covers the lane-1 late-tail lineage §2790–§2799 and sets the next hour. SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE
THE REAL MODEL — LOWER IS BETTER.

## Explained fraction (unchanged)

Strict ledger: 5.348% / 10.923% / 4.727 nat / 0 of 68 installed tail items. Nothing in §2790–§2799 installs into the §312 frontier
(norm-2304 at +2.6735). The hour produced structural facts about one channel, scored by CE on the FRESH split (docs 0–63;
baseline 3.0322401), with four of nine registered predictions falsified and preserved (§2798 b/c/d/e, §2799 b/d/e/f).

## Largest gaps

1. Tail dictionaries / coverage credit — zero credit, and the hour makes the reason exact: the late MLPs' 384-dim "tail" is a
   high-rank readout channel (§2798: PCA/W_U frames recover .18/.32/.62 and .11/.19/.49 of .1130 at k = 8/32/128; eff rank 261)
   with a full-width gate (§2799: 256 of 768 gate modes to get within .03). No dictionary, no small interface — description only.
2. The m16 remainder — untouched this hour.
3. attn5's write = the price cliff — untouched this hour.
4. NEW, honesty gap: the lineage has produced a precise description of the late MLPs' read (768-core quadratic + core-gated linear
   tail read, §2780–§2799) but no parameter accounting of what a program built from it would cost; a "smaller program" claim is
   not available until that is done (PR2).

## What the hour established (§2790–§2799)

The 384 dims outside the late MLPs' shared 768 read frame are written by the late MLPs themselves from their core (§2796, 73%
MLP(c)), accumulate rather than wire (§2790 recency), and are consumed mainly by the unembedding (§2797: readout .1130 / later MLPs
.0523 / attention .0031; readout marginal .0876; blocks 15–17 write 70% of what the readout reads). The channel is high-rank in the
readout's frame and in its own (§2798), and the gate that selects it is the whole core: the exact weight-side Gram of the cross
term has operator-family eff rank ≈ 600/768 and its data-weighted spectrum is the mean gate plus a long tail (§2799). Two prereg
design errors were recorded and scored as written (§2799 pred_e uncentred moment; pred_f reference). One process error (a rung
named after §2782's rung) was repaired bit-exact and guarded in derive.py.

## Candidates (tensor / polynomial / gauge / causal / program)

- C3 WHAT the readout reads from the tail (token classes; QUEUED now as late_tail_readout_content_probe): per-token FINAL_ONLY
  damage by induction/repeat/novel target (reusing Codex's ops/target_token_classes.py), frequency, baseline loss, position;
  and token-level Pearson between writers 15/16/17 — one shared channel or several specialised ones. Seven preds.
- PR2 honest parameter accounting of the late-read program (frames + folded weights vs the model; CPU, lane 2). Required before
  any "smaller" claim; also decides whether the lineage has any program value beyond description.
- T6 does the same 768 + tail split hold for the EARLY MLPs' consumers (who reads the early tail: §2783 found the structure; the
  consumer split was never measured early)? Cheap (parent consumer probe with SPLIT moved), tests whether the late description
  is a depth-general program item (compositionality across depth).
- P4 block 17 as a fixed linear tail map: §2799 says block 17's gate is rank-4 in energy. A rank-16 constant-gate surrogate of
  block 17's cross term is the one concrete compression the hour offered — small (one block), falsifiable (CE vs GATE_EXACT),
  would install nothing but would be the lineage's first parameter-reducing item.
- G2 gauge: is the 768/384 split itself canonical? Re-derive the core frame from the WRITE covariances (§2797's wheads) instead
  of the read covariances and check the tail is the same subspace (principal angles). Low cost, guards the whole lineage
  against a frame artefact.
- C4 attn5 price cliff revisit with the per-token instrument built for C3 (which tokens pay when attn5's write is denied) —
  reuses the new per-token scorer on the oldest open gap.

Prune: G2 is cheap and a real robustness check but low information if it passes (expected); P4 is small and concrete but
one block; T6 and C4 both reuse this hour's code; PR2 is bookkeeping with the highest honesty value.

## Ranked top five

1. C3 readout content by token class (executing — queued 02:29Z).
2. C4 attn5 per-token content with the same scorer — the oldest open gap, reused instrument, one afternoon of composition.
3. P4 block-17 constant-gate surrogate — the lineage's only compression candidate.
4. PR2 parameter accounting — before any program claim.
5. G2 frame canonicality — robustness of the split.

## Executed

C3 registered at 02:29Z (LATE_TAIL_READOUT_CONTENT_PROBE_PREREGISTRATION.md), derived from the §2797 consumer probe with
Codex's target_token_classes reused unchanged, smoked, gated, enqueued (lane 1). Next after it lands and is written as §2800:
C4 (attn5 per-token) or P4, by what C3 says about where the tail's value sits.
