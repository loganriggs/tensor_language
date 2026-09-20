# Shared radial feature and native-mean diagnostic

One new shared radial quadratic does not produce a qualifying simpler recent-path program. The native input distribution strongly violates the isotropic assumption used to set its coefficient. A calibration-fitted mean improves prediction, but still misses the10% gate. These are opened-panel results; neither is a discovered semantic circuit.

## Exact decomposition and implementation

Let S_k=sym(L_k outer R_k), t_k=tr(S_k), d=1152, and n=RMSNorm(h16). Then

B(n)=b + [sum_k D_k t_k/d] ||n||² + sum_k D_k n^T(S_k-t_k I/d)n.

The isotropic feature ||n||² is computed once and can feed every output. A retained subset K can equivalently execute its original products plus the omitted radial write [sum_(k not in K) D_k t_k/d]||n||². It preserves the full quadratic tensor trace, not its full function. RMS epsilon is explicit, including at zero input; ||n||² is never replaced by d.

The centered atom Gram is K_center=K-(C^T C) elementwise(t t^T)/d. Global and parent-reader metrics choose fixed supports; the random support matches earlier baselines. No polynomial tensor is materialized. Each compressed executor adds1152 radial writer values and1152 coordinate squares/summation to the retained-product cost. The radial feature is reused across output coordinates. Native attention17 and normalization remain, as do upstream h16/x0/v1 ports.

## Native comparison

Rows show maximum relative scalar/intervention-change error across scale0/.5/1/1.5. The first panel calibrates; the second is a separate previously opened validation panel. Both use8documents x64prediction positions.

| Radial coefficient | Support | Products | Calibration error | Validation error |
| --- | --- | ---: | ---: | ---: |
| isotropic weights | global_centered | 256 | 3.668 | 3.965 |
| isotropic weights | global_centered | 1024 | 3.550 | 3.822 |
| isotropic weights | global_centered | 2304 | 3.284 | 3.539 |
| isotropic weights | parent_centered | 0 | 3.919 | 4.269 |
| isotropic weights | parent_centered | 256 | 3.418 | 3.709 |
| isotropic weights | parent_centered | 1024 | 3.159 | 3.380 |
| isotropic weights | parent_centered | 2304 | 2.906 | 3.104 |
| isotropic weights | random_products | 256 | 3.336 | 3.657 |
| isotropic weights | random_products | 1024 | 2.415 | 2.634 |
| isotropic weights | random_products | 2304 | 1.131 | 1.222 |
| calibration mean | global_centered | 256 | 0.479 | 0.608 |
| calibration mean | global_centered | 1024 | 0.453 | 0.581 |
| calibration mean | global_centered | 2304 | 0.402 | 0.514 |
| calibration mean | parent_centered | 0 | 0.601 | 0.700 |
| calibration mean | parent_centered | 256 | 0.418 | 0.528 |
| calibration mean | parent_centered | 1024 | 0.377 | 0.474 |
| calibration mean | parent_centered | 2304 | 0.345 | 0.433 |
| calibration mean | random_products | 256 | 0.555 | 0.622 |
| calibration mean | random_products | 1024 | 0.472 | 0.492 |
| calibration mean | random_products | 2304 | 0.224 | 0.258 |

Both selectors return the dense reference. The best tested compressed calibration-mean arm uses2304random products and67% of dense conditional-program storage, but has26% validation error. A single random support is not a distribution over random baselines, and opened validation is not fresh confirmation.

## Red-team and interpretation

v635 keeps the v634 weight-defined supports but replaces t_k by the mean of (L_k n)(R_k n), estimated only on512calibration positions. Those4608estimated moments are fit statistics, not weights-only discovery. The radial parameterization preserves the fitted mean approximately on RMS-normalized inputs; native epsilon remains explicit. The fields named isotropic_support_* describe support selection only, not coefficient error of the fitted replacement.

The weight-implied isotropic write has norm4349.7, versus24266.2 for the observed calibration mean; their relative difference is1.0204. Thus preservation under isotropic inputs was not preservation under this native distribution. The fitted mean diagnostic improves errors markedly but cannot replace the remaining contextual variations. This is evidence about these panels, not a theorem that mean-aware sparse methods cannot work.

All three executor tests pass, including a radial feature independently expanded into coordinate squares, zero inputs and empty retained dictionaries. Four atom tests pass, including the centered Gram against explicit trace-free matrices. Full-support native replay passes. Each native experiment uses16forwards/256local replays and no model updates. v634 has no fitting; v635 estimates4608calibration moments.

## Consequence

Keep the new radial feature as a supported building block and a documented negative candidate. Do not interpret isotropic polynomial error as native functional fidelity. The next discovery step needs a richer shared quadratic feature dictionary or a carefully declared native-input metric; fitting a mean alone is insufficient. Any data-informed metric must be labeled as such and followed by genuinely fresh predictions, selective edits and reuse tests. Simple/OOD/extracted/selective/reusable circuit criteria remain unmet as a whole.

[v634 weights-only receipt](../bilinear_quotient/circuits/followups/recent_radial_feature_v634_result.json) · [v635 fit diagnostic](../bilinear_quotient/circuits/followups/recent_radial_feature_v635_result.json)
