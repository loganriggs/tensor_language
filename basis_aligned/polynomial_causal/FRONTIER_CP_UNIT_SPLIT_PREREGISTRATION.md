# Frontier: is the CP excess concentrated too? The unit-split analogue of §2919. Preregistration

Registered 2026-09-04T13:32Z (substituted by `date -u` at write time, so the stamp cannot drift from the value read). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## SIGN CONVENTION

Frontier L2 is **CE ADDED ABOVE THE REAL MODEL, so LOWER IS BETTER** (§2135; §312: "+2.6735 beating +2.84/+2.93"). A cost is
`L2(arm) − L2(baseline)`, **POSITIVE = WORSE**, so a **negative cost is an improvement**. §2128/§2129/§2133/§2134 RETRACTED;
**§2125 STANDS** — this rescales already-selected units; it does not reselect or reorder.

## Why

Four rungs have now said what the two adopted scalars are **not**:

| | | |
|---|---|---|
| §2917, §2920 | the tail scalar is not a mis-tuned ridge penalty | optimum 0.25 at λ×0.25, λ×1, λ×4 |
| §2918 | the CP scalar is not a truncation compensation | optimum 0.5 at keep 0.25, 0.5, 1.0 |
| §2919 | the tail excess is **not isotropic** | top 64 of 1152 directions read −0.2614 vs uniform −0.2287; the rest **hurts** (+0.0489) |

**The open question is whether the two corrections share a shape.** The CP units carry a natural importance ranking — the very one
`select_units` uses, `‖Dw[:,u]‖·‖L[u]‖·‖R[u]‖` — so "top r units" plays the role that "top r singular directions" played for the tail.
This runs §2919's experiment on the CP side: scale one end of that ranking by the **adopted 0.5**, leave the other at 1.0, and see
whether a subset beats the whole.

| | reading |
|---|---|
| **concentrated** (pred_d holds) | the CP correction is also a projection, and the two scalars share a structure **across completely different provenance** — ridge solutions on one side, the model's own untouched weights on the other |
| **spread** (pred_d fails) | the two look alike as scalars but are **not the same kind of object**, and the tail's low-rank structure is specific to fitted maps |

**Both are substantive.** A shared shape would be the strongest structural statement the campaign has; a clean separation would say the
"local objective" story and whatever governs the CP side are genuinely different mechanisms that happen to be correctable the same way.

## Predictions, each with its worked-example line

- **pred_a — REPRODUCTION GATE.** `|L2_F(baseline) − 2.6735| ≤ .05`. *Worked example:* +2.6735/+2.6736; past .05, nothing else reads.
- **pred_b — the split path at scale 1.0 is a physical no-op.** `|cost(all units × 1.0 through the split path)| ≤ .005`. *Worked
  example:* **0.0000**. **Registered up front because §2919 omitted exactly this control and it cost that rung its result** — its
  uniform arm went through a plain multiply, not the path its split arms used, so nothing proved the split path faithful. A failure
  here invalidates every other cell in this rung.
- **pred_c — both uniform routes reproduce §2902.** `|cost(all × 0.5 via the split path) − (−0.1074)| ≤ .01` **and** `|that − cost(plain
  multiply × 0.5)| ≤ .01`, with −0.1074 read from §2902's receipt at run time. *Worked example:* all three ≈ −0.1074. The route
  agreement is the sharper half: two code paths for the same operation must agree before any split arm is comparable to the adopted
  scalar.
- **pred_d — the CP excess is concentrated like the tail's.** `min_splits cost ≤ cost(uniform) − 0.01`. *Worked example:* concentrated ⇒
  some `top r` reads ≈ −0.12 to −0.14 against uniform's −0.1074, holds; spread ⇒ every `top r` is **worse** than uniform (each captures
  only part of a gain that needs all units) and it fails. **One-sided on purpose**: a split that merely ties uniform is not evidence of
  structure.
- **pred_e — the best split is interior to the rank grid.** *Worked example:* §2919's optimum sat at the **smallest rank it tested**, so
  its value is a bound; the grid here runs 64 → 1152 of 2304 units, and 128/2304 mirrors §2919's 64/1152 in fraction. **Registered
  because interiority is my recurring failure — §2907, §2909, §2917 and §2919 all failed or bounded on grids I chose too narrow.**

## Nulls

- **`b_null_the_split_path_is_not_faithful`** (> .02) — the rung is void, and I report it as void rather than reporting the arms.
- `c_null_the_uniform_routes_disagree` (> .02).
- `d_null_the_cp_excess_is_spread_across_units`.
- `e_null_the_rank_grid_is_too_narrow`.

**What I will do with each outcome, stated in advance.** pred_d holds with b, c ⇒ record that **both** adopted corrections are low-rank
projections rather than scalars, and the follow-up asks whether a joint projection composes with §2912 and survives a held-out window
(§2914/§2916) before anything is adopted. pred_d fails ⇒ record the **separation** as the finding: the CP correction genuinely needs all
its units and the tail's does not, which makes "one two-parameter object" (§2910) a coincidence of two different mechanisms rather than
one. **Nothing is adopted from this rung**; §2912 remains the frontier of record either way.

## Price

**1 full frontier pipeline run + 13 arms × 3 windows of forward evaluation, ≤ 600 GPU-seconds** (§2918's 16-arm run took 167.4 s; each
arm is a masked rescale of six `Dk` matrices), 0 backwards, **0 fitted parameters**. The parent `ops/frontier_fisher8.py` is
**unmodified**. Not forward-instrumented, so the receipt reports `gpu_forwards: 0` with `forwards_instrumented: false` and
`pipeline_runs: 1`. Receipt: `frontier_cp_unit_split_results.json`, read with `price` in the same command the ledger section is written
from, in the canonical `Price:` / `Results:` form (§2853, §2858), under a filename no other section cites (§2876).
