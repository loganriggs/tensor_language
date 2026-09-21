**The latest coefficient fits over-specialize to their calibration chunks**

This frozen comparison uses all 32 historical 64-token chunks. Seven chunks supplied the recent 32-coefficient fits; the other 25 were excluded from those particular fits. They are not fresh, certified document-disjoint, or untouched by earlier feature and metric construction. No model execution or fitting was performed.

| Generated component-three relative error | Seven fitted chunks | Other 25 historical chunks |
|---|---:|---:|
| Original graph | 8.871% | 7.702% |
| Paired-context refit | 7.320% | 8.027% |
| Cartesian-context refit | 7.843% | 8.108% |
| Covariance baseline | 12.048% | 8.548% |
| Isotropic baseline | 11.851% | 12.873% |

Both refits improve the fitting subset and worsen the remaining aggregate. Only 6/25 other chunks improve for the paired fit, and 8/25 for the Cartesian fit. However, deterioration is 4.21% and 5.27%, respectively: the registered prediction of at least 10% deterioration in both arms FAILS. This is evidence of specialization, not proof that it fully explains the fresh-panel failures. Original and refitted graphs still beat these baselines on this aggregate; that does not erase the previously observed subgroup failures.

**Successor: exact error along the coefficient update**

Only the second read changed. With the first read, normalization, and supplied carry fixed, the generated component is affine in the changed coefficients. Along the interpolation from original to refitted coefficients,

$$
E(t)=\|e_0+t\Delta f\|_2^2=a t^2+2bt+c.
$$

Here $t=0$ is the original graph and $t=1$ is the complete update. The CPU audit computes this quadratic from executed component values. It is an exact function-space interpolation under the declared fixed interface, not an assumption that arbitrary neural-network parameter interpolation is linear.

| Update | Best unconstrained step on fitted chunks | Best unconstrained step on other chunks |
|---|---:|---:|
| Paired | 1.709 | 0.175 |
| Cartesian | 1.623 | 0.047 |

Both directions initially reduce aggregate error on the other chunks, but their full fitted steps overshoot. The diagnostic best improvements on those chunks are tiny; these values are not used to select or export a new candidate. Steps beyond one are unconstrained diagnostics and need not preserve the original safeguards. This result motivates testing regularization selected within calibration, rather than changing graph topology solely because a full coefficient update failed. It does not establish OOD generalization or justify fitting to the opened evaluation panels.

The broader graph and full folded-target goals remain open. A generalization-controlled refit would need a genuine separation between coefficient selection and evaluation, with equally optimized baselines and document provenance. Historical chunk labels cannot supply that guarantee.

[Registered hypothesis](HISTORICAL_PREFIX_TRANSFER_PLAN_V1.json) · [Full per-chunk and interpolation results](HISTORICAL_PREFIX_TRANSFER_V1.json) · [Executable audit](audit_historical_prefix_transfer.py).
