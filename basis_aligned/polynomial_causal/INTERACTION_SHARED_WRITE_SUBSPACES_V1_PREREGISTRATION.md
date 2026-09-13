# Matched-budget shared-output-subspace fit

Frozen13September2026. Prior CPU operator controls pass; the32single-ray model remains unfinished near60%coefficient error. This fit relaxes each group to an eight-dimensional output subspace, retaining full input-product rank.

Object: all147,456twelve-output weight vectors from the head17.2/MLP17 mixed tensor. Each product receives one of32groups and eight signed coefficients; each group stores an orthonormal12×8output basis. Nominal FP32 storage is4,878,336bytes including uint8groups, close to the previous4,896,936byte sparse-entry10%error representation. No text fits or activation weighting.

Ten fixed random orthonormal starts receive20assignment/eigen updates. The strongest three receive up to500further updates or240seconds each. A small10-step objective change can trigger FP64 polishing; it is not convergence. Each promotion then receives up to20FP64updates after orthonormalizing its basis. At each frozen point, assignments/codes are optimal; exact group Gram eigensolves measure the total remaining conditional improvement divided by target squared norm. Two consecutive values at most1e-8certify the declared coordinate tolerance. Iteration or time limits without that certificate remain unfinished. This is not global convergence or stable semantic identification.

A: serialized FP32 coefficient loss differs from the FP64 fitted loss by at most1e-5; basis orthogonality error at most1e-5.
B: at least two promoted fits satisfy the FP64 coordinate certificate.
C: best serialized coefficient error at most10%and actual artifact bytes at most4,896,936.

All histories and misses retained. Save the best explicit program, even if its fidelity criterion fails. No behavioral or runtime adoption based on coefficient error alone. All input coordinate products and shared group features remain charged; this shares writes, not necessarily semantic tasks.

Managed lane1,900second cap; expected working memory below4GB. Source and weight-derived inputs bound in INTERACTION_SHARED_WRITE_SUBSPACES_V1_BINDING.json. Independent CPU conditional-gain identity and reassignment descent control precedes enqueue. Native/OOD/selective-intervention validation is a separate promotion after interpreting the fit.
