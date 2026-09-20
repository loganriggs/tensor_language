# Document-level covariance sensitivity — 2026-09-20 18:50 UTC

Before attributing covariance-weighted native gains to reusable structure, audit inputmetric stability. Savedpanelscontain32documents×64tokens, not2048independentdocuments. Within eachpanel, eight seeded disjoint16document splits measure covariance relative difference andmean alignment. Repeated splits overlap: descriptive sensitivity, not confidenceintervals or independentreplicates. Report singularity implied by1024centered rows in1152dimensions.

For unitdirections selected exclusively from calibration covariance (top16,bottom16,16seededrandom), compare normalized centered variance on the twofullpanels. The four-slot metric's penalty for a rank-one quartic is variance^4, so report that ratio too. No directionselection onpanel2, no model/student fitting. This may expose calibration anisotropy amplified bydegree; it cannot separate populationdistribution shift from finite/correlated sampling.

Pred_a_rows: reshape succeedswith32×64and covariancefromrows matchesstoredFP32stats<1e-6. Pred_b_sensitivity: medianwithinpanelhalf covariance relative difference>.5. Pred_c_amplification: somefixed direction's quartic penalty changes>4× or<1/4acrossfullpanels. Preservealloutcomes. Nativeweightedfit remainsfrozenqueued; this audit explains sensitivity withoutretuning ondiagnosticpanel.
