# Mathematical review — 2026-08-30 22:10 UTC

Convention: L2/gap = CE added above the real model; LOWER is better. Grounding: ledger §2135 (retraction) through
§2145 (tail attribution); rung 52 in flight.

## Executed this review (CPU, from existing receipts)

**1. The damage functional is ~1-additive over piece-grain pruning, with one measured large interaction.**
Möbius interaction I(A,B) = D(A∧B) − D(A) − D(B), fresh medians:

```
  I(c8@288, c9@288)              −0.0016
  I(halve mlp4/5, halve c6–c9)   −0.0062
  I(c6..c9 all to 288, 4-way)    +0.0053
  I(top8-selection, prune) cfgE  +0.0669   ← the one large term, and it crosses intervention KINDS
```

Within norm-rank pruning, interactions are ≤ 0.006 — the lattice is effectively 1-additive, which is why the
§2143 additive prediction landed to 0.0016 (§2144). The single large interaction (§2142) is between *selection*
and *pruning*, supporting a shared-error-budget model: interventions that drain the same amplified-noise channel
anti-compose. Operational rule adopted: registered pair predictions from singles are licensed within a pruning
kind; cross-kind pairs must be measured.

**2. Rate table (two-part-code / MDL view; corrected unit price 3,456 values per CP unit):**

```
  move                                ΔMvalues   Δfresh damage   rate (nat per Mvalue)
  c6–c9: 2304→576   (§2140)            −23.9        −0.0290         −0.0012  (free lunch)
  c8/c9: 576→288    (§2144)             −2.0        −0.0128         −0.0064  (best rate)
  leave attn16 real (§2145; rung 52)    −5.3        −0.157*         −0.030   (*prefix marginal; refit pending)
  mlp4/5: 2304→1728 (§2141)             −4.0        +0.0244         +0.0061  (never take)
```

Under the (stored values, damage) order, every accepted move is a strict Pareto improvement, and leave-attn16-real
— if rung 52 confirms the refit keeps most of the 0.157 — DOMINATES rather than trades: the a16L dictionary costs
storage AND damage. The only axis on which it is a retreat is *coverage* (components replaced), which is the
reverse-engineering objective, not the compression one; both ledgers must be kept.

This also produced the stored-value accounting correction now in the ledger (prices in §2139–§2144 undercounted
by ×1.8; damage numbers unaffected; the best config saves 25.9M values, not 14.4M).

## Ranked mathematical moves

1. **Sparse interaction models over the intervention lattice (executed above).** Object: the map from piece-grain
   configs to fresh damage. Definition: k-additive Möbius truncation. Assumption that may fail: additivity is
   kind-local (already falsified across kinds, §2142). Consequence beyond reconstruction: registered numeric
   predictions for unmeasured configs (twice confirmed at 0.002–0.006). Cheapest falsifier: rung 52's −0.10 bar.
2. **Class-conditional rank as the tail bottleneck (Hankel/automaton view).** Object: a16L's +0.157 vs a12L's
   +0.006. Definition: the aXL dictionary forces attention output through a rank-≤10 class-conditional mean map
   (+2 linear classes); its damage should be governed by the within-class residual energy of the real output.
   Assumption that may fail: damage is CE-weighted, not energy-weighted (the §2117 ρ = 0.81 says energy is a good
   but imperfect proxy). Consequence: predicts WHERE more classes/rank buy damage (spend rank at a16/a14, not
   a12/a17). Cheapest falsifier: rung 53 (below) — one capture pass, Spearman ρ bar.
3. **Two-part-code envelope as the program's scalar (executed above as the rate table).** Object: the whole
   frontier. Definition: minimize L(program) + n·D at exchange rate λ; with 1-additivity the argmin is the
   negative-rate set — solved greedily, no search. Assumption that may fail: additivity outside the measured
   lattice. Consequence: turns future rungs into rate measurements with an explicit stopping rule (take a move
   iff its rate < λ). Falsifier: any accepted-move combination whose joint damage misses the additive sum by
   > 0.01 (none so far except the known cross-kind pair).

Pruned: full Shapley estimation (2^n arms; 1-additive + measured pairs suffices); information-bottleneck variants
(duplicate the closed Fisher arc); invariant-theory gauge quotients (scale gauge already settled, §2113);
simultaneous/shared dictionaries across tail layers (would inherit the §2142 cross-kind interference and cannot
be certified per §2122's coverage rule without per-layer shares).

## Preregistered next (rung 53, backlog): the class-bottleneck spectrum test
For each tail layer li ∈ 10..17, within-class residual energy fraction e_li = E‖y_li − CV_li[c]‖² / E‖y_li‖² of
the REAL attention output under the deployed context (FW eval rows, oracle classes). pred_a Spearman
ρ(e_li, §2145 marginal_li) ≥ 0.7; pred_b argmax e = a16; pred_c the two smallest e are {a12, a17}. Null: damage
tracks CE-weighting, not energy (ρ < 0.7 with a16 still max would partially save the frame).
