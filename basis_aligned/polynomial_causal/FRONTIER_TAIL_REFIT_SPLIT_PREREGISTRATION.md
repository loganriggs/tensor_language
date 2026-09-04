# Frontier: inside the tail refits — class table or link maps? Preregistration

Registered 2026-09-04T09:32Z (`date -u` read in the same tool call that composed this header). Before the run. Immutable; the rung's
frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2_F(arm) − L2_F(baseline)`, **POSITIVE = WORSE**. §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS** — the frontier is norm-2304 at
2.6735.

## Why, and why the parameter split makes this worth a rung

§2878 established a dissociation: the sixteen `fit_attnd` dictionaries collapse for **0.0000 nats**, while the eight tail **refits**
`a10L`–`a17L` cost **+0.2011** to collapse — about half of this pipeline's whole tail-attention increment of +0.3864. It did not ask
where inside a refit that cost lives, and the parameter split makes the question unusually valuable because it is wildly asymmetric:

| piece | parameters per layer | share |
|---|---|---|
| class table (ten rows) | 10 × 1152 = **11,520** | 0.22% |
| link maps (LINK = [2,7,8,9]) | 4 × 1152 × 1152 = **5,308,416** | **99.78%** |

If the +0.2011 is carried by the **class table**, the link maps — **42,467,328 parameters across the eight tail layers** — can be
dropped at no cost while the frontier keeps everything they were buying. If it is carried by the **link maps**, the tail dictionaries
are genuinely expensive and this line of simplification stops.

Three arms of §312's published norm-selection pipeline, applied at the **inline tail-refit site only**. The `fit_attnd` site is
deliberately left untouched — §2876 measured it free at 0.0000, so leaving it alone isolates the refits and makes pred_d's accounting
against §2878 meaningful. §2878's joint cost is read from its receipt **under a frozen hash** rather than retyped.

| arm | change |
|---|---|
| BASELINE | none |
| links dropped | `LW := {}` — keep the class table |
| table collapsed | `CV` ← ten copies of `Y.mean(0)` — keep the link maps |

Derived from `ops/frontier_fisher8.py` (§2125 rung 30), which is **unmodified**; the derived file retargets the parent's single `OUT`,
and §2878's receipt is preserved under a distinct name after the overwrite incident disclosed in §2876.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE, carried over verbatim from §2125 rung 30.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:*
  §2874–§2878 all read +2.6735/+2.6736 on this derivation; past .05 and **nothing else here is readable.**
- **pred_b — the link maps are free.** `cost_links ≤ +.05` nats. *Worked example:* if the class identity is what the tail dictionary
  contributes and the within-class linear correction is decoration, ≈ **+.00–.03**, and 42.5M parameters leave the construction at no
  cost; if the linear maps are the mechanism, ≈ **+.15** and `b_null_the_link_maps_are_load_bearing` fires.
- **pred_c — the class table carries the cost.** `cost_table ≥ +.15` nats. *Worked example:* §2878's joint cost was +.2011, so if the
  table carries it this reads ≈ **+.17–.20**; if the table is free (≈ **+.02**) then neither piece alone explains §2878 and the cost is
  interactional, which pred_d would then expose as a large drift.
- **pred_d — the two pieces account for §2878.** `|cost_links + cost_table − 0.2011| ≤ .05`. *Worked example:* clean additive
  decomposition reads ≈ **.01**; a drift of **.1+** means the two pieces interact and neither arm can be read as "the cost of that
  piece". This is the clause that keeps pred_b and pred_c honest, and it is checked against §2878's receipt under a frozen hash so the
  comparison cannot drift from the number it is compared to.
- **pred_e — the parameter split is stated exactly**: 11,520 vs 5,308,416 per layer, 42,467,328 for the eight tail layers' link maps.

## Nulls

- `b_null_the_link_maps_are_load_bearing` (`cost_links ≥ +.15`): the 99.78% of parameters are doing the work, and the tail
  dictionaries cannot be cheapened this way — the outcome that closes this line and is registered so it is recognised.
- `c_null_the_class_table_is_free` (`cost_table ≤ +.05`): the ten rows are decoration; combined with a passing pred_b it would mean
  **neither** piece alone explains §2878 and the cost is interactional.

## Price

**3 full frontier pipeline runs, ≤ 800 GPU-seconds** (§2874–§2878 measured 279–283 s for three arms), 0 backwards, 0 fitted parameters
beyond the pipeline's own. The parent is **not forward-instrumented**, so the receipt reports `gpu_forwards: 0` with
`forwards_instrumented: false` and `pipeline_runs: 3` beside it, and the ledger's `Price:` line says so — the count is absent, not zero.
Receipt: `frontier_tail_refit_split_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form (§2853, §2858), under a filename **no other section
cites** (§2876).
