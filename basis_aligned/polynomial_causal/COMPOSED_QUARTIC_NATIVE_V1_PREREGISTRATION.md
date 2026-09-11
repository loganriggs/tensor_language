# Native full-unembedding quartic contraction and sampling price

11September2026, before native execution. Builds the exact homogeneous producer/producer route U→MLP17→MLP16. Uses all native weights, scaled MLP16 output with actual lambda17_0, no biases in the homogeneous degree4 term. All lower-degree bias terms, attention/background contributions and RMS denominators remain external and are not claimed explained.

CPU symmetric tensor, exhaustive coefficient norm and factor-gradient controls held within4.53e-15. Existing joint_router_polynomial_gram_v1 already supplied dense quartic symmetrization; new contribution is factorized whole-two-MLP contraction and its full-output coefficient sampling objective.

For four independent vectors x1..x4 of identity covariance, E||T(x1,x2,x3,x4)||²=||T||F². Use4096Gaussian and4096Rademacher probes, separate deterministic seeds11411/11412, batches64, FP64 full-U output metric. No text or learned activation weighting. This is a native oracle/price test, not a sparse fit or circuit result.

A: bound CPUcontrols; native diagonal contraction equals sequential bias-free composition relative<=1e-10 and input-slot permutation invariance<=1e-10; finite positive estimates.
B: estimated relative standard error<=5% for both probe distributions, reporting actual independent sample count.
C: Gaussian/Rademacher norm estimates differ by<=3sqrt(SEg²+SEr²). These are estimated sampling errors, not rigorous probability bounds. A B/C miss calls for variance/sample controls, not a model-structure negative.

Also report the norm of one pair-partition coefficient tensor before full four-slot symmetrization, and its ratio to the fully symmetric tensor norm, descriptively. Symmetrization is an orthogonal coefficient projection; a reduction is not automatically learned structure or a sparse fit. No native product-port loss is substituted for the composed quartic loss.

Zero body forwards;8192 synthetic coefficient probes, batch64; max300seconds, no persistent input/weight tensor. Record forward seconds per probe distribution and peak GPU memory. Native routing and full model behavior are not evaluated. Queue throughlane1 behind existing matched sparse path fit. This establishes whether the proposed deeper-object fitting loss is affordable; subsequent sparse fitting and four-property validation remain required.
