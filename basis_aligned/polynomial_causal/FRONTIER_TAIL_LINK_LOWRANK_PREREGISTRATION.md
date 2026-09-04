# Frontier: a low-rank tail link map — the last cheap-simplification lead. Preregistration

Registered 2026-09-04T09:44Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2_F(arm) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — norm-2304 at 2.6735.

## Why this is the only lead left on the tail side

§2881 closed the obvious route and left exactly one open:

| piece | parameters / layer | cost to remove |
|---|---|---|
| four 1152×1152 within-class link maps | 5,308,416 (99.78%) | **+0.1740** |
| ten-row class table | 11,520 (0.22%) | +0.0248 |

So the maps are the **mechanism**, not decoration — and per parameter the table is ≈ **6,600×** more efficient, which says the maps are
**wasteful rather than useless**. A wasteful full-rank map is exactly what a low-rank one replaces.

Three arms: BASELINE, every `LW[k]` truncated to **rank 8**, and to **rank 1**. A rank-r map costs `2·r·1152` numbers instead of
`1152²`, so across four LINK classes and eight tail layers **rank 8 is 73,728 parameters against 42,467,328 — a 576× compression** with
**0.1740 nats** of headroom to recover.

§2881's full-drop cost is read from its receipt **under a frozen hash** rather than retyped, so the recovered-fraction arithmetic cannot
drift from the number it is measured against. Per §2879's standing rule the manipulated entries are the `a10L`–`a17L` members of
`order2`, which §2881 showed move L2_F by .174; **pred_d re-checks that here as a measured predicate** rather than assuming it.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* every rung in this
  family reads +2.6735/+2.6736; past .05 and **nothing else here is readable.**
- **pred_b — rank 8 recovers most of the link maps' value.** `cost_rank8 ≤ +.05` nats, i.e. recovers ≥ **71%** of §2881's +0.1740.
  *Worked example:* if the fitted maps are low-rank in substance, rank 8 costs ≈ **+.01–.03** and 42.4M parameters compress 576× for
  almost nothing; if they are genuinely full-rank, rank 8 costs ≈ **+.15**, barely better than deleting them, which is
  `b_null_low_rank_does_not_recover`.
- **pred_c — rank 1 is worse than rank 8.** `cost_rank1 − cost_rank8 ≥ +.02`. *Worked example:* a monotonicity sanity bound — more rank
  should not hurt — giving ≈ **+.05**; if rank makes no difference (≤ 0) then the truncation is not doing what the arm name says and
  neither number should be read, which is `c_null_rank_does_not_matter`.
- **pred_d — the arms are connected.** `|cost_rank1| ≥ .005`. *Worked example:* §2879's rule as a measured predicate — a disconnected
  manipulation reads exactly **.0000**, as `fit_attnd` did three times. Rank 1 is the most aggressive arm here, so if even it reads
  .0000 the truncation never reached the evaluated config and nothing in this rung is readable.
- **pred_e — the parameter accounting is stated exactly**: 42,467,328 full / 73,728 at rank 8 / 9,216 at rank 1, and the 576×
  compression figure.

## Nulls

- `b_null_low_rank_does_not_recover` (`cost_rank8 ≥ +.12`, i.e. recovers < 31%): the tail link maps are substantively full-rank, and
  **there is no cheap tail dictionary by any route this ledger has tried.** That closes the tail side entirely and is worth the price
  as a clean negative.
- `c_null_rank_does_not_matter` (`cost_rank1 − cost_rank8 ≤ 0`): the truncation is not behaving monotonically and the rung is void.

## Price

**3 full frontier pipeline runs, ≤ 900 GPU-seconds** (this family measures 279–283 s for three arms; the added SVDs are 32 matrices of
1152×1152, negligible), 0 backwards, 0 fitted parameters beyond the pipeline's own. The parent is **not forward-instrumented**, so the
receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line
says so — the count is absent, not zero. Receipt: `frontier_tail_link_lowrank_results.json`, read with `price` in the same command the
ledger section is written from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858),
under a filename no other section cites (§2876).
