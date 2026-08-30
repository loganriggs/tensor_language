# Hourly strategic review — 2026-08-30 18:35 UTC (self-reviewed; Claude, single lane)

## 1. Since the 17:35 review

Fourteen registered runs landed through `bqrunner` (lane 1 §2101–§2114 plus the three price/cliff artifacts);
every one has a ledger entry, a result JSON, and its failures scored as written. Synthesis:
`explanations/explanation_1835.md`. Two instrument facts were caught by the gate/runner and preserved (a
duplicate binding; a per-block detach that cut the autograd graph; K = 4,608 being the whole MLP width).

The arc, in one line each:
- Random stream error is priced by depth with a cliff **at attn5's write** (0.075 → 1.72 nat per half-norm);
  the cost is local; scale is free.
- The certified arm's own error is **anti-random** (2.4× a matched random error at block 6); its observable
  third carries the price (oracle-correcting it recovers 94.5 %).
- By head: the sink head 5.7 is **71 % of the block-6 stream error and 19 % of the CE**; the other eight attn5
  heads are 23 % and 85 %. Rel-MSE, the benchmark's currency since §311, is wrong by ~4× at the cliff.
- **Certified frontier gain:** mlp4/mlp5 units selected by the block-5/6 loss-gradient metric —
  +0.124 / +0.075 nat on two document-disjoint windows at equal price; **metric-1152 = norm-2304** (half the
  price at equal CE); the selector reduces to **eight named directions** per site (newline-vs-place-name,
  markup/punctuation structure, place names), 8 × 1152 stored values.

## 2. Fraction explained — unchanged strict ledger

5.348 % certified removable storage · 10.923 % of deletion CE named · 4.727 nat unnamed · 0/68 complete
terminal circuits. The gain above improves a *stand-in* at equal price; it does not yet move a certified
whole-model number (the +2.93 fresh-pile frontier is a different window family and not re-certified).

## 3. Largest remaining gaps

1. **The frontier number is priced in the wrong currency.** Every registry priority computed as
   `delta_opt × (1 − fidelity)` on rel-MSE inherits the ~4× head-grain distortion measured in §2114.
2. **The gain is certified on two windows of one token file**, not on the eight fresh-pile windows of §2083.
3. **Half of the CE gap to the uncompressed mlp4/mlp5 is capacity, not choice** (§2107): selection closes 52 %;
   the rest needs a different program for those layers.
4. **m16's per-document code stays private** (§2100); the causal-response route is closed.

## 4. Candidates, pruned

| candidate | gain | causal/composable | falsifiable | GPU | redundant |
|---|---|---|---|---|---|
| A. Ladder the eight non-sink attn5 heads by price (running) | closes the head-grain story | yes | yes | 1 min | no |
| B. Certify the metric-unit gain on §2083's eight fresh-pile windows | makes it quotable | yes | yes | ~10 min + pile-10k download | no |
| C. Re-price the theseus registry in CE-at-the-cliff (per-component oracle correction inside the arm, ordered by CE not rel-MSE) | reorders priorities | yes | yes | ~15 min | no |
| D. Eight-direction selector for the *front tables'* residual bases (rank-64 → chosen in the top-8 metric) | small (bases −0.009 under full metric) | — | yes | 3 min | mostly closed by §2105 |
| E. m16 as a measured two-number interface | fixed cost | yes | yes | hours | not now |

**Ranked:** A, B, C, E, D.

## 5. Action executed

A is on the runner (`attn5_head_price_ladder.py`, preregistered: top-2 heads ≥ 60 % of the other-eight recovery;
top head is h5 by the eight; singles additive within 30 %; reproduction gate on §2114's +0.713). B is the next
queued item once A lands (it needs the pile-10k windows of `ops/probe_gate7.py`; the download is a known cost).
Nothing here is an outcome until its artifact exists.

## 6. Blockers

None external. Standing: self-reviewed; Codex lane off by decision.
