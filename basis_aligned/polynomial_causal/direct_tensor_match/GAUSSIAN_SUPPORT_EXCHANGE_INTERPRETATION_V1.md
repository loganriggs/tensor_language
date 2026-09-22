Mixed-metric graph edits: small objective gains, component failure

22 September2026,00:10UTC. Managed native run completed58.23s, peak2.58GB allocated. Both frozen144-by4quadratic producer dictionaries; same768candidate pool as coefficient-only experiment,512activepairs with144protecteddiagonals. Sixteen accepted exchanges perstart, exact outputreadout refitted. Deployment stays1088products,1353728floats,1024indices.

Registered integrityPASS; learningPASS (captured score gains0.3814%/0.1450%, each above0.1%, and both text errors decrease); componentFAIL. No criterion changed after evaluation.

| Metric | Start1101 before→after | Start1102 before→after |
|---|---:|---:|
|Text16-output error|21.057→20.375%|20.003→19.842%|
|Root1 same-token response error|36.597→36.118%|38.147→38.990%|
|Root1 sensitivity error|40.130→39.213%|42.413→42.270%|
|Sampled coefficient error|98.886→98.792%|99.417→99.352%|
|Isotropic Gaussian error|87.327→87.585%|88.709→88.756%|

The fitting Gaussian uses empirical mean/covariance. The last row uses independent isotropic Gaussian probes and is a different metric; it worsens slightly. The fixed mixture multiplier is34445.84/27000.23. Holding these multipliers fixed does not maintain the original coefficient constraint after changing support.

Baseline readout replay<1.6e-8, normal equations<1.6e-15, physicalFP32export<1.4e-7. These support instrument correctness, not semantic interpretation.

Solving with all768rootproducts yields captured scores only0.4124%/0.3243% above the final512supports. This is an upperbound on remaining captured mixed-score improvement from support selection within this fixedpool, with the same ridge and featureproducers. It is not an upperbound on component fidelity. The actual fullpool fit has text20.107/19.655%, response36.412/38.862%, sensitivity38.900/41.628%; it does not repair components either.

Contrast with the CPU evaluation oracle: the same fullpool can reach sensitivity8.00/8.77% while enforcing exact cached pairresponses if fitted on evaluation labels. Thus this negative result does not establish inadequate fullpool capacity. The mixed objective and fixed producers do not select a readout with those properties. Calibration-label fits also transfer poorly, so replacing the loss with a small empirical fit is not a demonstrated solution.

Decision: further bounded greedy support search with the same objective/pool is lower priority. The most useful next comparison moves the shared quadratic input factors, analogous to the mixedCP feature-learning experiment, while preserving the1088-product architecture. A new streamed exact-gradient helper passes five planted algebra/gradient checks; native memory/runtime must be measured before committing to a long schedule. No shared-family adoption, OOD result or semantic circuit is established.

Artifacts: [preregistration](GAUSSIAN_SUPPORT_EXCHANGE_PLAN_V1.md), [native results](GAUSSIAN_SUPPORT_EXCHANGE_NATIVE_V1.json), [pool capacity](SHARED_POOL_RESPONSE_CAPACITY_INTERPRETATION_V1.md), [transfer geometry](SHARED_TRANSFER_GEOMETRY_INTERPRETATION_V1.md), [gradient controls](SHARED_MIXED_GRADIENT_CONTROLS_V1.json).
