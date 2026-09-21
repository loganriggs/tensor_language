**Longer exact CP fitting improves coefficient matching, but functional fidelity remains poor.**

Two random starts,400Muon steps with a stretched cosine schedule,1536products/2385920coefficients, same16fixed data-informed output readers. Runtime259.65s, integrity PASS, learning PASS, joint comparison to prior hierarchy FAIL.

|Start|Sampled coefficient error|Gaussian function error|Opened text scalar error|
|---|---:|---:|---:|
|1001|97.776%|98.976%|40.875%|
|1002|97.899%|98.878%|39.027%|
|Earlier exact shared hierarchy|98.475%|96.355%|57.700%|
|Inherited empirical shared hierarchy|not measured here|147.091%|8.131%|

Both CP fits improve on their25step pilots and beat the earlier exact hierarchy in sampled coefficient/text errors, but lose on Gaussian function error. Costs and initialization differ; this is not an equal-budget architecture comparison. None is faithful globally or adopted for interventions. Sampled coefficient entries do not certify full Frobenius error. The exact regularized explained score reaches.005182/.005184 versus earlier hierarchy.003316, but teacher constant remains omitted and these are not relative errors.

**Actual CPU geometry audit changes the interpretation.** Both CP programs retain numerical input-span rank1152. Yet the top256 directions contain95.2–95.3%of the unweighted factor-direction Gram trace, up from49.6%after25steps. Coefficient-weighted direction summaries agree. Thus nominal full-span capacity coexists with highly concentrated learned directions; this is descriptive, gauge-dependent geometry, not a proof that the target itself is low-rank or that all remaining directions are useless.

The two learned polynomial tensors have exact coefficient cosine.8143, compared with.000582 after25steps. This is partial functional agreement, not stable feature identification: their difference still has60.96%of the first program's norm. One start has a pair of atoms with absolute coefficient cosine>.99; most atoms are not identical. Diagonal individual-atom energies omit cancellation and cannot alone justify pruning.

**Next representation control is genuinely shared.** A bank of144quadratic features, each4products, with512explicitly selected feature pairs costs1088products/1353728coefficients/1024indices and permits1152input directions. This keeps reuse while avoiding the complete10440-pair dictionary. Seeded pair support is a structural assumption, not discovered sparsity.

Five dense value/cross/gradient controls pass below6e-15. Known sparse-bank recovery with4features/7pairs,400steps and two starts yielded Adam1/10 andMuon1/10 below1%; widening to8features/24pairs while retaining teacher support yielded Adam7/10 andMuon10/10, medians.4505%/.01550%. More parameters aid optimization even though the narrower planted target was exactly representable. These are different teachers from the CP suite, so no cross-suite optimizer ranking is implied.

The next native fit should compare this sparse shared bank against CP and old hierarchy, using exact weight loss and separately reporting coefficient, Gaussian and text metrics. Recovering planted supports does not guarantee randomly seeded native support is suitable. Full-model reconstruction, stable meanings, selective manipulation and reusable causal adoption remain open.

[Native results](QUARTIC_CP512_NATIVE_V2.json) · [Exact program geometry](QUARTIC_CP512_GEOMETRY_V2.json) · [Sparse-bank exact controls](SPARSE_QUARTIC_BANK_CONTROLS_V1.json) · [Wider planted recovery](SPARSE_QUARTIC_BANK_RECOVERY_WIDE_V1.json).
