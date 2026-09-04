# Frontier: is the front excess concentrated too? The third object type. Preregistration

Registered 2026-09-04T13:44Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-fitted components; it neither selects nor reorders.

## Why

Two of the three correction sites are now known to be **concentrated, at opposite ends of their rankings**:

| site | provenance | where the excess lives | best | vs uniform |
|---|---|---|---|---|
| tail `LW` | ridge solutions | **top 32 of 1152** singular directions | −0.2828 | −0.2287 (§2921) |
| CP `Dk` | the model's own weights | **all but the top 128** of 2304 units | −0.1280 | −0.1074 (§2922) |

**The third site has never been asked.** §2895 found the front tables' quadratic residual `A` wants ×0.5, worth **−0.1648** standalone —
the second-largest single correction in the campaign — and one that **failed to compose**: §2904 measured TCF at −0.2885, *worse* than
TC's −0.3213. Each installed `tableres` carries `A` (1680×64) lifted by `P` (1152×64), so its **64 components** have a natural
importance `‖A[:,j]‖·‖P[:,j]‖`, and the same split experiment applies directly.

**Why this deserves a rung rather than a footnote.** A scalar front correction does not compose; a *concentrated* one might. The usual
reason a scalar fails in composition is that it damages directions the other terms rely on — which is precisely what §2921 found for the
tail, where shrinking the bottom of the spectrum **hurt** by +0.0489. **If the front excess is concentrated, §2904's composition failure
has both a candidate explanation and a candidate fix.**

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735; past .05, nothing else reads.
- **pred_b — the split path at scale 1.0 is a physical no-op.** `|cost(all components × 1.0 through the split path)| ≤ .005`. *Worked
  example:* 0.0000, as §2922's CP identity read. **Registered up front**; §2919's omission of this control cost that rung its result.
- **pred_c — both uniform routes reproduce §2895.** `|cost(all × 0.5 via the split path) − (−0.1648)| ≤ .01` **and** `|that −
  cost(plain multiply × 0.5)| ≤ .01`, with −0.1648 read from §2904's receipt at run time. *Worked example:* all three ≈ −0.1648.
- **pred_d — the front excess is concentrated.** `min_splits cost ≤ cost(uniform) − 0.01`. *Worked example:* concentrated ⇒ some split
  reads ≈ −0.18 to −0.20 against uniform's −0.1648, holds; spread ⇒ every split is worse than uniform because each captures only part of
  a gain that needs all 64 components, and it fails. **One-sided on purpose**: a tie is not evidence of structure.
- **pred_e — the best split is interior to the rank grid.** **Computed on the winning arm's rank irrespective of which half wins.**
  *Worked example:* this is the **fix registered in §2922**, where my check admitted only `top` winners and so became unsatisfiable the
  moment the CP excess turned out to live at the bottom — the predicate failed on my coding, not on the grid. Here the grid is
  {4, 8, 16, 32, 48} of 64, so an optimum at 8, 16 or 32 is interior either way. A non-interior optimum makes the value a **bound**.

## Nulls

- `b_null_the_split_path_is_not_faithful` (> .02) — the rung is void and reported as void.
- `c_null_the_uniform_routes_disagree` (> .02).
- `d_null_the_front_excess_is_spread_across_components` — a real outcome: it would say the front site is genuinely different from the
  other two, and that "corrections are concentrated" is a property of two sites rather than a principle.
- `e_null_the_rank_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** pred_d holds with b and c ⇒ record that **all three** correction sites are
concentrated, and register a composition rung asking whether a concentrated front term composes where §2895's scalar did not — which
would reopen a line §2904 closed, on new grounds and with the reason for the original failure identified. pred_d fails ⇒ record the
front site as the exception and say plainly that concentration is not universal. **Nothing is adopted from this rung**: §2923's
configuration (rank-32 tail projection, CP ×0.80, motif ×1.25, **+2.2732 / +2.2953 held out**) remains the frontier of record, and any
front term must earn its place through a composed, held-out measurement as §2923 did.

## Price

**1 full frontier pipeline run + 13 arms × 3 windows of forward evaluation, ≤ 600 GPU-seconds** (§2922's 13-arm run took 158.5 s; each
arm is a masked rescale of three `A` matrices), 0 backwards, **0 fitted parameters**. The parent `ops/frontier_fisher8.py` is
**unmodified**. Not forward-instrumented, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and
`pipeline_runs: 1`. Only the **installed** `tableres` entries are touched — `_F0` filters on `k in order2`, which is the §2879 check
that a manipulated component is actually in the config. Receipt: `frontier_front_component_split_results.json`, read with `price` in the
same command the ledger section is written from, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other
section cites (§2876).
