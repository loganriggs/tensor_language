# How much of the original interaction does the current program explain?

13 September2026. The generated-background/full-cross program preserves its local intervention well, but it is not yet an accurate account of the original two-edit interaction. Its regional target-effect errors on that larger target are11.9–16.1%. An executed receipt decomposition locates most of the missing contribution before block10.

## Aligned targets and an exact telescope

The native full-cross effects in the newest generated-background receipt exactly equal those in the earlier17-arm GPU experiment, across all160prefixes and both endpoints. This checked equality allows their measurements to be combined without assuming two independently executed references match.

Write $f_N,f_C,f_R,f_P$ for final readouts under no removal, child removal, remainder removal and their joint parent removal. The original interaction is

$$
I=f_P-f_C-f_R+f_N.
$$

Let $f_A$ be the old arm4: make the post-MLP9 state additive, then execute block10 and the remaining suffix. Let $f_B$ be arm5: make the post-MLP10 state additive, then execute the suffix. Exactly,

$$
I=(f_P-f_A)+(f_A-f_B)+(f_B-f_C-f_R+f_N).
$$

These terms represent, along this specified intervention path, propagation of the pre-block10 mixed state, block10 mixed generation, and suffix nonadditivity from an additive post-block10 state. They are telescoping endpoint differences, not unique or independent causal shares of the model.

The current candidate predicts a parent readout by applying the native suffix to its generated additive post-MLP10 background plus its generated direct cross-product. Its predicted interaction subtracts the native child/remainder readouts only for scoring. Those readouts are not inputs to the parent-state generator.

## Coverage on the four regional groups

Signed projection means $\langle X,I\rangle/\|I\|^2$. It describes alignment with this original interaction vector; it is not an explained-variance percentage.

| Group | Current target error | Current control error | Pre-block10 signed projection | Local cross signed projection | Suffix baseline signed projection |
|---|---:|---:|---:|---:|---:|
|0|11.93%|38.88%|10.39%|6.59%|82.42%|
|1|12.16%|38.74%|11.56%|14.78%|73.40%|
|2|16.05%|30.57%|15.45%|28.09%|56.21%|
|3|14.57%|29.44%|13.84%|18.55%|67.05%|

The difference between the entire block10 term and its direct cross-product has only0.25–0.60% signed projection on the regional target. The current approximation captures much of the original interaction through the retained native suffix/background calculation, not solely through the explicitly compressed cross-product. This is why tiny error on the local-product test must not be advertised as tiny error on the original behavior or as a transparent explanation of that whole behavior.

FineWeb original-target errors range11.0–57.2%. The earlier local FineWeb criterion and these larger-target errors are different questions; neither passes universally.

## Executed discriminator and next computation

As a diagnostic only, add the actual endpoint difference $f_P-f_A$ to the current prediction. Regional target errors fall to0.48–0.86%; control errors to2.35–3.93%. This supports the pre-block10 contribution as the main next missing term. The correction uses the true parent output and is therefore an oracle diagnosis, not a legal repair or an extracted predictor.

A legal route already has the necessary mathematical ingredient. For the common head9 writer, the exact normalized MLP9 response obeys

$$
\Delta(a+b)=\Delta(a)+\Delta(b)+\mu_9(a,b),
$$

where $\mu_9$ lies in the two-vector mixed-response basis derived in RESPONSE_PRODUCT_BASIS_V2_MATH.md. Instead of assuming the joint post-MLP9 state is additive, pass the generated $\Delta(a+b)$ through the same attention10 and MLP10 maps. This keeps the inherited mixed term and its interactions with both branches. Then evaluate the generated joint, child and remainder states consistently.

This requires no new learned parameters, but adds a third generated branch and retains the dense attention/MLP weights. Its fidelity and literal execution price must be measured. The result would advance conditional extraction/composition; it would not by itself produce a small autonomous model, fresh language OOD evidence, or stable semantic identification. The new target is the original joint-removal behavior, not another improvement to the already accurate local cross-only target.

[Coverage and source hashes](COMPOSED_ORIGINAL_INTERACTION_COVERAGE_V1_RESULT.json) · [Executed oracle-gap diagnostic](COMPOSED_ORIGINAL_INTERACTION_ORACLE_GAP_V1_RESULT.json) · [Reproducible receipt analysis](composed_original_interaction_coverage_v1.py).
