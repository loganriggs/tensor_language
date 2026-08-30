# Full bilin18 update since `explanation_1745.md` — 18:35 UTC

**Date:** 2026-08-30  
**Coverage:** the observability arc from its first brick to a certified, priced, and mechanistically located
frontier improvement (lane 1 §2101–§2114, `STREAM_ERROR_PRICE_V1_RESULT.md`, `PRICE_CLIFF_SUBLAYER_V1_RESULT.md`).
Fifty minutes of runner time; fourteen registered runs; every number below has a preserved artifact.  
**Primary rule:** a plan or a queued runner is not an outcome. All self-reviewed (no auditor on this instance).

## 1. The honest short answer

The strict whole-model totals are unchanged (5.348 % certified removable storage; 10.923 % of deletion CE named;
4.727 nat unnamed; 0/68 complete terminal circuits). What changed is that the program now has, for the first time
since 14:05, a **certified improvement of its frontier assembly** with a **named mechanism** and a **priced
compression**, and it has measured — not argued — that the currency it had been pricing stand-ins in (stream
error) is the wrong one.

## 2. How error is priced in bilin18 (the empirical error budget)

- **A cliff, not a ramp.** A random perturbation of half the stream's norm costs 0.02–0.06 nat at the inputs of
  blocks 0–5 and **1.48 nat at block 6**, peaks at block 7 (1.81), and decays to ~0.4 by block 17. The jump happens
  inside block 5, **at attn5's residual write** (0.075 → 1.72 nat), not at mlp5 (adds nothing) — the last
  gatherer-band attention commits the per-position stream.
- **The cost is local.** Perturbing one position after block 5 hurts that position's prediction (0.82 nat) and
  barely touches later ones (0.13 summed): the tail cannot repair it and does not spread it.
- **Scale is free; direction is not.** Rescaling the stream by 1.5 costs ≤ 0.06 nat anywhere before the last
  block. The first-order observable subspace is two-thirds of the stream at every depth, so "factor only the linear
  quotient" buys ≤ ⅓ — but a program's *own* error is anti-random: the certified arm's block-6 error costs 2.4× a
  random error of the same norm, and oracle-correcting only its observable third recovers 94.5 % of what full
  correction recovers.

## 3. Where the certified assembly's error comes from, and what it costs

Lane 1's certified empirical arm (all attention real; front MLP tables, CP middles, tail spans compressed) carries
rel-MSE 0.51 at block 1 rising to 1.74 at block 6, and a +1.5-nat gap on its evaluation rows.

- **No single front piece is the lever** (m0 is fifth of seven); **attn5 amplifies mlp4's error 8.6×**; and across
  seven one-piece-real arms, block-6 rel-MSE does not predict CE (ρ = 0.07).
- **By head:** 74 % of the error attn5 injects is the §1089 sink/bias head 5.7's (97 % of the layer's real output)
  — a mis-scaled fixed vector lying off the loss-gradient directions at a random direction's rate. Oracle-correcting
  it removes **71 % of the block-6 stream error and 19 % of the CE**; correcting the other eight heads removes
  **23 % of the stream error and 85 % of the CE**. Correcting all of attn5 recovers 0.79 of the 1.50-nat gap.
  Zero-ablating head 7 is not an option: it costs the real model 0.91 nat.

This is the measured statement that rel-MSE — the benchmark's pricing currency since §311 — is wrong by a factor of
~4 at the cliff.

## 4. The improvement, its mechanism, and its price

- **Gain (§2105–§2106, certified on two document-disjoint windows):** selecting which 2,304 of mlp4's and mlp5's
  4,608 units to keep by importance under the block-5/6 loss-gradient metric instead of output norm buys
  **+0.124 nat (window 1) and +0.075 nat (window 2) at identical stored values**. The residual bases of the tables
  contribute nothing (−0.009); a random-metric control gives 0.017; extending the selector to c6–c9 lowers stream
  error and *raises* CE.
- **Price (§2107):** metric selection at K = 1,152 matches norm selection at K = 2,304 on both windows —
  **half the stored values at equal CE**. The gain grows as capacity shrinks (0.156 / 0.124 / 0 at K = 1152 / 2304 /
  full). One registered bar failed because K = 4,608 is the whole layer — an instrument fact missed at
  registration, scored as written.
- **Mechanism (§2108–§2111):** the gain is not about the 90 % observable span (every unit has ~68 % of its energy
  there); it is the Gramian's **top eight directions** per site. A selector built from just those eight reproduces
  the gain (+0.129 / +0.065) and beats a top-87 selector: **8 × 1152 stored values replace a Gramian.** The eight,
  named by the tokens whose block-5 stream loads on them: one dominant **newline-vs-place-name** direction carrying
  18 % of the loss's sensitivity at block 5 by itself; five structure directions (parentheses, markdown headers,
  list punctuation, currency/URL fragments); two place-name directions. They are not unembedding directions
  (overlap 0.10) and not the massive activation. attn5's sink head reads them at 3.7× random through its second
  bilinear factor; the block-5 and block-6 eights overlap 0.47.

In words: the front's compressed mlp4/mlp5 were keeping loud units that write where the loss does not look;
choosing instead the units that write into the eight structure/place-name directions attn5 reads — where the loss
is least forgiving — is worth an eighth of a nat at equal price, or half the price at equal CE.

## 5. What the mathematics contributed

- **Prospective splits with registered nulls** caught three would-be over-readings (source-side low rank is
  generic; a two-thirds subspace is not "small"; recovery is concave so a random half-space recovers 62 %).
- **Gramian eigenstructure as a selector** is a spectral statement: ||G^{1/2} w|| is dominated by G's steep head
  (r50 = 87 of 1152 at block 5), which is why eight directions suffice and why widening to 87 hurts.
- **A metric does not move an unconstrained least-squares fit**, so the gain could only come from constrained
  choices (unit sets, residual bases) — predicted before the split run, confirmed by it (bases −0.009).
- **Decomposition beats ablation for load-bearing components:** zero-ablation of the sink head measures the head,
  not the amplifier; per-head differences on shared positions measure the injection.

## 6. Blockers and confusing results

None external. Two standing limitations: everything is self-reviewed, and the certified gain is certified on two
windows of one token file, not on the fresh-pile windows of §2083. The confusing result is the sink head: it is at
once the largest source of stream error, nearly free to the loss in that error, load-bearing (0.91 nat), and the
strongest reader of the eight directions that price everything else.

## 7. Current plan, in order

1. **Ladder the eight non-sink attn5 heads by price** (`attn5_head_price_ladder.py`, queued): which heads carry
   the 85 %.
2. **Certify on the §2083 fresh-pile windows** before quoting the +0.124 / half-price result as a frontier move.
3. **Re-price the benchmark's registry entries in CE-at-the-cliff, not rel-MSE** — the head-grain factor of ~4 is
   large enough to reorder priorities.
4. **The m16 interface** (§2098–§2100) remains a private two-number-per-document code; unchanged.
5. A whole-model composition test only for a survivor.

## 8. Primary artifacts

- lane 1 ledger §2101–§2114; `BENCHMARK_BACKLOG.md` rungs 11–21
- `STREAM_ERROR_PRICE_V1_RESULT.md`, `PRICE_CLIFF_SUBLAYER_V1_RESULT.md`, `OBSERVABILITY_QUOTIENT_V1_RESULT.md`
- result JSONs: `assembly_error_quotient`, `observable_correction`, `front_piece_amplification`,
  `metric_front_refit{,_split}`, `metric_units_{certify,ksweep,mechanism,mechanism2,top8}`, `name_the_eight`,
  `head7_amplifier`, `attn5_error_by_head`, `head_energy_vs_price` (all under `basis_aligned/bilinear_quotient/`)

## UPDATE — 18:55 UTC (§2116–§2119)

- **Certified on the eight fresh pile-10k windows of §2083:** the equal-price gain holds 8/8 (median +0.082 nat,
  largest on the hardest text), and the **eight-direction selector** itself certifies at 7/8 with the same median.
- **Withdrawn:** the "half the price at equal CE" statement in §4 above. On the eight windows, metric-selected units
  at K = 1,152 are *worse* than norm-selected at 2,304 on six windows (median −0.028); §2107's two-window result
  reproduced exactly and did not transfer. No compression factor is certified.
- **Corrected:** "rel-MSE is wrong by ~4×" holds *inside attn5* (the sink head); across the assembly's thirteen
  compressed pieces, own-output error energy ranks CE recovery at ρ = 0.81 (§2117). The registry's rel-MSE
  priorities are not licensed to be re-ordered by the head-grain result.

## UPDATE 2 — 19:35 UTC (§2125–§2127)

- **The certified selector does not install into the §312 frontier** (§2125): published +2.6735 reproduced
  exactly; the fisher8 arm is worse (−0.048 fresh, −0.025 C) with matched-context refits throughout. Reading:
  the eight were computed for the *real* readers; the frontier's deployed readers are dictionaries. The gain
  stands certified as cfgE-specific; rung 32 (assembly-conditioned Fisher) is the falsifiable repair.
- **The sink-head scalar closes negative** (§2126): s* = 1.095 — cfgE *under*-drives head 5.7, the opposite
  sign to §1818's compiled-program 159× — and buys 0.015 of the head's 0.28-nat oracle. Per-head scale
  corrections are not a lever; the scale error belongs to the front's stream.
- **The m16 two-number interface fails** (§2127): held-out R² 0.069 against a 0.5 bar — m16's per-document
  response varies in shape, not two amplitudes. No cheap measured interface; the block stays the priced
  unexplained remainder.
