# Conditional normalized MLP11 response fold

The 27-writer mixed attention edge now folds through the joint normalized MLP11 response. All eight opened-v4 composition cells pass the registered 10% number / 5% control thresholds, measured against the weaker singleton effect. Worst output error is 0.000267823 (0.0268%). Native baseline replay is exact; local FP64 response relative error is at most 4.45e-12. This is a context-prepared path, not an independently extracted or identified circuit.

For background h, writer matrix W, coordinates z and P(h)=Down[(Left h)*(Right h)], define s(h)=mean(h²)+eps. The residual response is

    Wz + [P(h+Wz)-P(h) - P(h)/s(h)*(s(h+Wz)-s(h))]/s(h+Wz).

Contract jointly into 27 linear and 378 packed quadratic coefficients per output; retain the exact quadratic normalization denominator. Both orders of off-diagonal products are included. Bias cancels. Zero intervention produces exactly zero response. Native singleton backgrounds and suffix blocks12–17 remain required and charged; frozen per-input selectors remain oracle-dependent. V4 is opened repair evidence.

Primary native receipt: ../bilinear_quotient/circuits/followups/v4_mlp_response_fold_v1_result.json. Implementation: prepared_bilinear_response_tensor.py. Preregistration: V4_MLP_RESPONSE_FOLD_V1_PREREGISTRATION.md. Export: ../bilinear_quotient/circuits/followups/v4_mlp_response_fold_v1_program.pt (first row of first context only).

Independent CPU replay uses explicit loop-built monomials and matrix products: maximum absolute error 6.60e-12, relative error 1.30e-12, exact zero response. See MLP_RESPONSE_EXPORT_CPU_V1.json. Planted normalized-constant write cancellation and random direct secant checks are in PREPARED_BILINEAR_RESPONSE_CPU_V1.json. These controls defend algebra, not global causal interpretation.

## Baseline comparison

Canonical prepared representation costs 498,070 scalars/context, including carry and denominator. The conditioned native factors cost 290,710/context plus 5,308,416 shared Down weights. For 48 contexts, canonical costs 23,907,360 versus 19,262,496 for implicit factors. Canonical expansion therefore does not win this storage baseline. Preparation cost and native background generation are additional, not free. Sparse Tucker/HT on this new object have not yet been run.

## Next circuit screen, executed CPU consequence

On the single exported context at its frozen coordinate, removing the numerator's quadratic terms changes the local residual response by 0.0972%; removing the linear terms changes it by 48.66%. Their cosine is -0.9403. This is a conditional degree split, not a semantic feature or proof of behaviorally dispensable quadratics. The exact denominator remains in both tests. Native all-cell output testing is required before adopting the simpler numerator; small state error can be amplified downstream. Existing mixed-edge-alone causal insufficiency remains valid: this construction also uses the joint normalized MLP and native suffix.

## Native degree-removal and carry-only controls

Follow-up v4_mlp_degree_split_v1 passes both degree omissions8/8: linear-only worst number error0.0322%, quadratic-only2.9107%. The prediction that the linear response is necessary fails. Follow-up v4_mlp_carry_control_v1 also passes8/8 with the whole response numerator removed: number2.9104%, controls1.5067%. Both instruments pass, native anchors0. Carry-only adds the mixed edge to the post-MLP additive-background state; native background MLP and suffix remain. This is a simpler conditional composition route, not removal of the native MLP or a claim that MLP curvature is generally dispensable. Exactfold improves precision to0.0241%in this latest run.

The direct linear-only compiler prepared_linear_mlp_response.py avoids the quadratic output core, stores62614values/context, and matches the masked full compiler to2.84e-14 on independent random CPU tests. It is a precision baseline; carry-only requires no additional response coefficients at these gates. New sparse Tucker fitting is not justified before testing this cheaper behavioral baseline's amplitude/context transfer.

Native receipts: ../bilinear_quotient/circuits/followups/v4_mlp_degree_split_v1_result.json and v4_mlp_carry_control_v1_result.json. Preregistrations and independent CPU receipt are linked by matching V4_MLP_DEGREE_SPLIT_V1, V4_MLP_CARRY_CONTROL_V1 and PREPARED_LINEAR_MLP_RESPONSE_CPU_V1 stems. Preserve the old six-MLP zero-quadratic failure: that is a different conditional interface.
