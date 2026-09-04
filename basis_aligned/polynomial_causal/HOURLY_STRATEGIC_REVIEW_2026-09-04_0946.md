# Hourly strategic review — 2026-09-04 09:46Z (Claude, lane 1)

## Where the program stands

**Explained fraction (strict ledger): 5.348% / 10.923% / 4.727 nat / 0 of 68 — UNCHANGED, all session.**
Largest gaps, unchanged: **tail dictionaries / coverage credit**; the **m16 remainder**; **attn5's write = the price cliff**.

SIGN CONVENTION throughout (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating
+2.84/+2.93); a cfgE "gap" is damage and a cfgE "gain" is gap reduction. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — Fisher
selection does not install into the §312 frontier, which is norm-2304 at 2.6735.

## The hour in one paragraph, including what I got wrong

The pivot to the frontier produced real measurements and one serious self-inflicted detour. §2874/§2875/§2876 reported 0.0000-nat
"simplifications" of `fit_attnd` dictionaries; **§2879 withdrew all three** — those dictionaries are not in the evaluated configuration
(`order2 = cfgF + ['a10L'…'a17L']`, `cfgF = ['a0','m0E','a1v','m1','m2E','m3E'] + ['c4'…'c9'] + ['tailE']`; attention 2–9 enters through
the 38 motif heads). The independent physical control was a measurement, not the code reading: the identical manipulation moved L2_F by
**+0.2011** at the tail-refit site and by **0.0000** at the `fit_attnd` site, in an instrument whose resolution §2876 measured as
**0.0 at four decimals**.

## What survives, and is worth building on

- **§2878/§2881 — the tail dictionaries are installed, cost +0.2011 to collapse, and the cost tracks the parameters.** The four
  1152×1152 link maps are 99.78% of the weights and carry **87.5%** of the cost; the ten-row class table is 0.22% and carries 12.5%.
  The two pieces sum to +0.1988 against §2878's independent +0.2011 — **drift .0023**, a genuine cross-validation.
- **§2877/§2880 — the MLP stage does not reduce, and is strongly superadditive.** Each half costs ~0.7 nats alone; both together
  **+3.2104** against an additive prediction of 1.4350. **Single-component ablation understates that stage by 2.2×** — the Hydra
  effect (arXiv:2307.15771) inside a *fitted reconstruction*, and a caution for every per-component cost this ledger has published from
  single ablations.
- **§2876 — the pipeline is deterministic to four decimals.** This is what makes every other number here sharp, and it is what turned a
  run of exact zeroes from "free" into "disconnected".

## Candidates, pruned by information gain / falsifiability / GPU cost / redundancy

1. **attn5's error share inside the frontier, via `ML`.** Attacks a *named largest gap* that has stood for weeks on model-side evidence
   alone. `ML` is passed straight into `evalM`, so it is **installed by construction** — the §2879 rule satisfied by design rather than
   by luck. Falsifiable both ways with a control chosen to be hard (a2, §2834's second-largest in the band at 0.349 nats against a5's
   2.211). ~280 GPU-s. **RANK 1 — executed.**
2. **Cross-side additivity (attention error vs MLP error).** The two decomposition rungs in flight test additivity *within* each side;
   nothing tests it *across*. §2880 showed superadditivity is real in this construction, so the cross term may be large — and if it is,
   every block-wise share is an underestimate. **RANK 2 — blocked until both decompositions land**, then cheap.
3. **Low-rank tail link maps.** The last cheap-simplification lead: 576× compression with .174 nats of headroom. **RANK 3 — already
   queued** (prereg 09:43Z).
4. **Re-pricing the ledger's published per-component costs jointly.** §2880's caution generalised. Highest value per *claim* corrected,
   but combinatorial in GPU cost and the two decompositions will already indicate whether it is needed. **RANK 4 — deferred as
   premature.**
5. **The m16 remainder.** A named largest gap, but `m16` is not in `cfgF` and I have not established which construction it belongs to;
   scoping it is a reading task, not a rung. **RANK 5 — blocked on scoping, and I will not guess after §2879.**

Pruned without ranking: anything on the CLOSED list (v1 factorization, m16 cheap interface §2127, sink-head scalar §2126, c6–c9
**reordering** §2131 — note §2131 closed reordering, not costing, and the queued MLP-side rung prices the block without reordering it;
metric-constructed bases; half-price/K-reduction §2118; conditioning on cfgE §2132). Also pruned: further circuit-battery work — §2871
and §2872 established that instrument cannot resolve per-component selectivity, and nothing since has changed that.

## Executed: rank 1

`ops/frontier_attn5_error_share.py`, preregistration `FRONTIER_ATTN5_ERROR_SHARE_PREREGISTRATION.md` (09:46Z), **enqueued**.

Three arms of §312's published norm-selection pipeline: BASELINE (`ML = [2…9]`), attention 5 restored to real (`ML` minus 5), attention
2 restored as control. The motif heads are an approximation, so dropping a layer **restores the real component** and should **lower**
L2 — the quantity is an **error share**, `L2_F(baseline) − L2_F(restored)`, **POSITIVE = that layer's approximation costs that much**.

- **pred_a** reproduction gate, verbatim from §2125 rung 30 (|L2_F − 2.6735| ≤ .05) — without it nothing else is readable.
- **pred_b** `share_a5 ≥ +.15` (worked example: §2830 put attn5 3rd of 36 and 20.4× disproportionate per unit written, so ≈ +.2–.6 if
  that transfers; ≈ .00 if the motif heads already approximate it well).
- **pred_c** `share_a5 ≥ 3 × share_a2` — specificity against a deliberately hard control.
- **pred_d** both arms connected (`|share| ≥ .005`): §2879's rule as a *measured* predicate, since a disconnected arm reads exactly
  .0000.
- **pred_e** restoring a layer never harms (`share ≥ −.01`); a materially negative share would mean the approximation beats real
  attention in this stack — a surprise worth its own section, not a bug.
- Nulls: `b_null_attn5_is_not_a_frontier_error_source` (≤ .03) and `c_null_attn5_is_not_special_in_the_band` (ratio ≤ 1.5). **Either
  firing closes the price-cliff gap on the frontier side**, which after weeks of model-side-only evidence is worth the price.
- Price: 3 pipeline runs, ≤ 900 GPU-seconds; receipt `frontier_attn5_error_share_results.json` under a filename no other section cites.

## Process changes adopted this hour

- **§2879:** before reporting a frontier cost, verify the manipulated entry is in `order2`/`cfgF` **and state that it is**. A run of
  identical exact zeroes across manipulations of very different sizes means the instrument is disconnected, not that the component is
  free.
- **§2876:** two ledger sections must never cite one receipt — a re-run overwrites it and destroys the earlier section's evidence.
  `ops/audit_ledger_prices.py` now refuses that, and reports unauditable sections instead of skipping them silently.

## Queue

Depth 3: `frontier_mlp_side_error_share`, `frontier_tail_link_lowrank`, `frontier_attn5_error_share`; `frontier_error_decomposition`
running.
