**Shared root blocks save storage and arithmetic, but the primary narrowly misses native fidelity.**

The parent is the preregistered joint-value/derivative program withlambda1. Its32 quadratic features are retained and computed once. Each root block reads a linear combination of their pair products and shares one output direction. Diagonalizing that quadratic form gives an explicit program of linear combinations, signed weighted squares and an output readout. This is a restricted arithmetic graph, not a general HT-to-DAG search.

| Output-shared forms | Products | Stored coefficients | Additions | Parent approximation error, panel0 / panel1 | Native quartic error, panel0 / panel1 |
|---|---:|---:|---:|---:|---:|
|Parent|656|903,168|901,856|0 / 0|7.74% / 13.61%|
|4|256|303,744|302,300|6.46% / 5.59%|9.78% / 14.05%|
|8, primary|384|312,576|311,000|4.11% / 3.54%|8.60% / 13.70%|
|16|640|330,240|328,400|2.90% / 2.44%|8.13% / 13.56%|
|32|1,152|365,568|363,200|2.27% / 1.94%|7.96% / 13.54%|

Primary8 saves41.46%of products,65.39%of stored coefficients and65.51%of additions. Both the dense leaf projections and the physical residual writer are charged. The unembedding remains a common external operation. Scalar coefficient applications are represented by stored coefficients; product counts denote products of computed variables.

Integrity PASS: projected-root replay<5.7e-15, physicalFP32export<5.1e-7, price formulas match. Parent approximation<=5%both PASS. Native error<=1.1timesparent both FAIL: calibration8.599%exceeds8.519%, despite second-panel13.703%being close to parent13.609%. Keep this narrow failure; do not promote a secondary rank after looking at the results. The16-form secondary meets that value ratio but barely saves products and remains only a diagnostic comparison. Runtime9.52s.

The graph's root space was chosen from parent calibration predictions. That does not minimize the joint value/derivative objective under an output-rank constraint. A successor exact reduced-rank solver has now been implemented and independently checked on five structures against an augmented-design projection oracle. For a fixed dictionary and positive-definite regularized feature Gram G, minimize ||C G^(1/2)-X G^(-1/2)||F at fixed output rank by truncated SVD in whitened coordinates. This permits fitting the limited output space directly to native joint targets rather than truncating an approximation by a different metric. It does not guarantee improved natural-input accuracy or solve feature discovery.

No native derivative test or full normalized-model intervention is claimed for these block programs yet. The pure-quartic target excludes residual/bias paths and intervening normalization. The learned scalar features and output-shared forms are not identified semantic units. A smaller approximate parent is useful compression evidence, not circuit adoption.

[Managed screen](SHARED_ROOT_BLOCK_NATIVE_V1.json) · [Wide compiler controls](SHARED_ROOT_BLOCK_SHAPE_V1.json) · [Reduced-rank solver controls](QUARTIC_RANK_READOUT_CONTROLS_V1.json) · [Parent fit](QUARTIC_JOINT_READOUT_INTERPRETATION_V1.md).
