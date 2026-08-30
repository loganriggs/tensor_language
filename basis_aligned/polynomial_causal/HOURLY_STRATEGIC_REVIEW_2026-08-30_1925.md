# Hourly strategic review — 2026-08-30 19:25 UTC (self-reviewed; Claude, single lane)

## 1. Since the 18:35 review

Twelve more registered runs landed (§2115–§2124 plus the math-lane `fisher_metric_v1`), every one written up,
scored as written, and pushed. **The observability arc closed at rungs 11–29.** Its balance:

- **Certified, LABEL-FREE, equal-price frontier gain:** mlp4/mlp5 CP units ranked by the top-8 loss-gradient —
  or, equivalently in effect, the true-Fisher-with-model-sampled-labels — directions at blocks 5/6 beat norm
  selection on the eight fresh pile-10k windows (8/8 full metric, 7/8 empirical top-8, 8/8 true-Fisher top-8;
  medians +0.082 / +0.082 / +0.086 nat; up to +0.22 on hard text). The eight directions are named (§2111).
  Deploy status: weights + unlabeled inputs, same as fold tables.
- **Withdrawn / corrected, before being quoted anywhere but the ledger:** the half-price claim (§2118: worse on
  6/8 fresh windows); "rel-MSE is the wrong currency" scoped to attn5's interior (piece-grain ρ = 0.81, §2117);
  the tail-span "gains" exposed as coverage artifacts (ρ = −0.976, §2122) and the coverage-stated credit for
  projection stand-ins licensed; §2106's "window 2 untouched" corrected (tail spans' PCA uses those rows).
- **Head-grain anatomy of the certified arm:** the sink head 5.7 is 71 % of the block-6 stream error and 19 % of
  the CE; four heads (5, 0, 1, 6) carry the price additively; correcting all of attn5 recovers 0.79 of the gap.
- **Mathematical identity (§2123):** the eight are half label-dependent, site-local (no first-order transport,
  pullback 0.40), super-quadratic at the cliff; the Fisher's *scale* prices small errors at 0.58× at both sites.
  Yet the *selector* built from the label-free true Fisher performs identically (§2124) — the unit ranking is
  insensitive to which half of the span moves.

## 2. Fraction explained — strict ledger unchanged

5.348 % / 10.923 % / 4.727 nat / 0 of 68. The certified gain improves the assembly's stand-ins at equal price;
whether it moves the *quotable frontier number* is exactly what is on the GPU now.

## 3. Largest gaps

1. The certified gain is proven on cfgE; the +2.67/+2.93 frontier configs carry motif-head and tail-dictionary
   error the selector does not touch — composition may dilute it (running).
2. The registry's tail entries over-credit projection stand-ins (coverage; §2122) — bookkeeping owed.
3. m16's per-document code remains private (§2098–§2100); the m16 interface is unbuilt.
4. The eight are site-local: every new site needs its own Gramian (cheap, but not one object).

## 4. Candidates, pruned and ranked

1. **(RUNNING) Install the selector into the §312 frontier** (`ops/frontier_fisher8.py`, rung 30): rerun the
   empirical-L2 pipeline with true-Fisher top-8 selection at mlp4/mlp5; reproduction gate on the published
   2.6735; gain bar 0.04; no-harm bar on window C.
2. **Coverage-credit the registry** (rung 31, CPU): annotate `registry/priorities.md` and the theseus tail
   entries with covered-energy shares from §2122's construction; re-derive the priority ordering.
3. **m16 as a measured two-number interface** (priced: 2 of 98 arms per document).
4. **The sink-head scale repair inside cfgE** (§1818's reciprocal, now aimed by §2113's decomposition).
5. Whole-model composition only for a survivor.

Pruned: anything re-running closed rungs; metric-constructed bases/spans (twice negative); c6–c9 selection;
half-price restatements.

## 5. Action executed

Rung 30 was preregistered and queued before this review and is mid-run (ARM 1/2, the reproduction arm). Its
result will be written as §2125 when it lands; nothing here counts it. Next in the queue after it: rung 31
(coverage-credit bookkeeping) unless rung 30's outcome reorders the list.

## 6. Blockers

None external. Self-reviewed throughout; Codex lane off by design.
