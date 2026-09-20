# Shared source quadratics and a coverage correction

The preceding context-specific signed rank-one Hessian baseline passed unit, negative and mixed source-amplitude tests. This did not identify a shared circuit feature. We now test both shared coefficient dictionaries and a stronger amplitude design.

## Shared tensor object and split

T[c,i,j]=H_c[i,j] has192 context/site output slots and3 named source-amplitude input coordinates. The input slots receive the same amplitude vector. Each H is symmetric. Vectorize the six distinct entries with off-diagonals multiplied by sqrt(2), preserving Frobenius inner products. This avoids a silent gauge/metric change when comparing tensor decompositions.

Fit fixed two-feature output SVD only to near_greeted/outside_called coefficient tensors (96 contexts); score yesterday_helped/earlier_noticed (96 contexts). Outcomes were already opened, so this is construction-held-out coefficient fitting, not prospective fresh OOD. The resulting two shared quadratic forms have context-specific mixing coefficients. No behavioral labels were fitted. Compare a matched two-feature random dictionary, common input-rank2 HOSVD, and retaining three of six entries per shared core. Sparse support uses coefficient magnitudes; indices are priced.

The registered shared-output2 held-out prediction gate fails: mixed edits16.59% versus10%. It beats the random control, whose worst non-double error23.91% is larger. Input-rank2 also fails mixed12.92%. Sparse-core shared2 fails17.14% on mixed and worsens unit/negative. Literal storage: shared-output2 twelve shared numbers plus five/context; input-rank2 six shared plus six/context; native full quadratic nine/context. Producers still require native Hessians. These reductions are not causal adoption or token-model compression.

## Red-team: previous amplitude design did not span the tensor

The four earlier arms a=(1,1,1),(-1,-1,-1),(1,-1,1),(2,2,2) span only two quadratic outer-product directions. Negative and doubled amplitudes do not add independent Hessian measurements. Consequently these probes cannot validate a general3x3 symmetric quadratic form, even when their scalar predictions pass.

The previously measured native source census supplies six independent settings e1,e2,e3,e1+e2,e1+e3,e2+e3. Their symmetric quadratic design has rank6, the full dimension. This is not a theorem that the true normalized model is quadratic; it is a meaningful coverage check for the proposed quadratic approximation. The CPU audit reuses those native receipts, with no new model calls or fitting.

| Method | Worst own-effect-relative error across six settings |
|---|---:|
| Full quadratic |4.72%|
| Per-context signed rank1 |15.74%|
| Shared-output2 |17.33%|
| Shared-input2 |18.29%|

The full quadratic passes the10% bar; all compressed alternatives fail. Per-context rank1 error normalized by combined B effect is9.99585%, but that larger denominator does not rescue the own-effect-relative failure. No rounding or denominator substitution is allowed. The earlier restricted-arm success remains accurate with its narrower scope; any general rank-one circuit claim is rejected.

## Evidence and continuation

[Shared dictionary result](SHARED_SOURCE_QUADRATICS_V1_RESULT.json), [six-direction audit](FULL_SPAN_SOURCE_QUADRATIC_AUDIT.json), and their executors test_shared_source_quadratics.py and audit_full_span_source_quadratics.py contain all cell results, designs, coefficients and prices. Isometric symmetric packing and exact full-basis reconstruction pass1e-12.

Use rank of the symmetric monomial design as a required coverage diagnostic for future quadratic circuit screens. It complements signed amplitude, norm, and native-capability tests; it does not replace any of them. The next live question is whether source observables can be generated more cheaply while preserving full-span responses and collateral outputs. Increasing dictionary rank until it reconstructs is not stable circuit identification. Native source and derivative generators remain open; the program goal remains unachieved.
