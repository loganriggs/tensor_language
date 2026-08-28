# Shared-QK-384 complete standalone program: predictive pass, context-gate failure

Date: 2026-08-28

Status: measured gate failure. Four of five preregistered gate families pass; the causal
context-transport gate fails. The candidate is not promoted as the first admitted
simplified complete program.

## What passes

The checkpoint-independent program stores 490,165,686 float32 values, saving 55,738,368
values (10.2103%) from the exact dense reference. It has total token support, zero fitted
lookup tables, zero native calls or module references, disjoint storage, and is scored
after the checkpoint object is destroyed.

Predictive composition is strong and replicates:

| role | all-position CE harm | covered harm | unseen-current harm |
|---|---:|---:|---:|
| skip7000 | +0.018433 | +0.019083 | +0.016387 |
| skip11000 | +0.019906 | +0.021248 | +0.015969 |

Covered CE replays the parent attention-bank result within $4\times10^{-8}$ nat. The
role-to-role covered-harm difference is 0.00216 nat. Thus the old attention-stake result
does compose through the complete owned shell, and unseen current tokens are not the
source of its error.

## What fails

On the frozen prefix intervention:

- native maximum downstream logit change: 3.497423;
- compressed maximum downstream logit change: 3.347464;
- delta norm ratio: 0.961304;
- delta cosine: 0.921108, below the 0.95 gate;
- context-delta recovery: 0.846824, below the 0.90 gate.

The compressed model preserves most effect magnitude but rotates the full downstream
effect vector enough to fail. This is not visible in its approximately 0.02-nat CE harm.
It is direct evidence that predictive reconstruction and causal interchangeability are
distinct simplicity consequences.

## Consequence

Strict simplified whole-model admission remains at zero under the frozen standard. The
candidate is nevertheless the first complete, independently executable measured
compression point and defines a real three-axis frontier: 10.21% storage saving,
approximately 0.02-nat predictive harm, and 84.68% context-delta recovery.

The cheapest discriminating next experiment is shared-QK rank 512 under the identical
whole-program gate. It saves 42,467,328 values (7.78% of the dense program). If it clears
context recovery, rank is the limiting resource; if not, the activation-MSE routing
basis is misaligned with causal transport and needs a context-weighted objective.
