# Target-feasible DAS regularization red-team — 2026-09-07

## Question

Did noise/Jacobian regularization uncover a meaningfully different solution from the same-rank,
same-initialization unregularized fit, or did the optimizer's improvement over difference in means
come from the target-feasible DAS objective itself?

## Complete 30-configuration comparison

All 30 configurations are target-feasible in both parity directions. Within each fixed rank and
initialization, regularization changes the cross-fold selection score only slightly:

| rank | initialization | no-reg score | best regularized score | change | no-reg / best-reg mean A1 | no-reg / best-reg mean P KL |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DIM task SVD | .146864 | .147040 | +.000176 | .825482 / .825460 | .016693 / .016695 |
| 1 | factor SVD | .132714 | .132217 | -.000497 | .872407 / .871786 | .010698 / .010258 |
| 2 | DIM task SVD | .155468 | .155660 | +.000192 | .870685 / .870982 | .013826 / .013889 |
| 2 | factor SVD | .182251 | .182294 | +.000043 | .871568 / .871196 | .011733 / .011584 |
| 4 | DIM task SVD | .208849 | .208435 | -.000414 | .825573 / .826529 | .015044 / .014854 |
| 4 | factor SVD | .188809 | .188564 | -.000245 | .879712 / .880127 | .017296 / .017307 |

Here “best regularized” means the lowest selection score among the four registered nonzero
noise/Jacobian settings, and negative change is better. The selected rank-one factor-SVD setting
reduces mean P KL by `.000440` while reducing mean A1 projection by `.000621`; its maximum P KL
drops from `.034585` to `.033000`, and both versions retain the same one P flip. A different
registered setting (`sigma=.10`, Jacobian weight `1.0`) has a slightly lower score than no-reg
`.132587` and higher mean A1 `.874598`, but still the same one P flip. These are perturbations of
the same solution family, not a qualitative regularization effect.

Rank and initialization matter far more. Factor-SVD rank one beats DIM-task-SVD rank one by about
`.014` in the selection score. Ranks two and four are less stable or more collateral despite their
capacity: rank-two factor-SVD minimum principal cosine is only `.147-.164`, and rank-four
DIM-task-SVD is `.022-.089`, versus about `.84-.86` for rank-one fits. The optimizer's real gain over
the external DIM baseline is therefore attributable to the target-feasible causal objective and a
good causal-factor initialization, not to the tested noise/Jacobian regularizers.

## Verdict and next discriminating observation

The user's overfitting diagnosis is supported at the construction level but not repaired by these
local regularizers. The fit generalizes across row parity and restart geometry while failing a new
construction, which is exactly what one expects when the missing information is environment
variation rather than local smoothness. The next test should add complete construction environments
to fitting and hold out another construction. A larger coefficient grid cannot discriminate that
hypothesis. Weight-tensor translation can meanwhile report which fixed readers align with the
selected direction, but those alignments remain diagnostic until construction transfer passes.

