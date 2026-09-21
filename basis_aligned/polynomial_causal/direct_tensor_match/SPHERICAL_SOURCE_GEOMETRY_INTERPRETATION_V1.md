**Passing the artificial-input predictions did not validate the fitting distribution.**

The full MLP16 weight-generated source diagnostic completed in1.144s. All three registered predictions passed: sampled source moments agree with the exact sphere calculation, mixed fitting beats unguarded learned fitting, and isotropic fitting beats pruning on all three artificial carry scales. The last result contradicts native candidate ordering: isotropic fitting has substantially worse native full-path errors than pruning and mixed fitting. Do not optimize this artificial law as if its fidelity transferred to native circuits.

| Independent carry/source RMS ratio | Pruned response error | Learned response error | Mixed response error | Isotropic response error |
|---|---:|---:|---:|---:|
|0.5|29.68%|42.87%|29.16%|28.07%|
|1.0|28.20%|41.82%|27.77%|26.61%|
|2.0|22.56%|34.83%|22.24%|21.28%|

These are centered full-vocabulary source-swap response errors after explicit normalization and softcap. Natural replacement errors are also in the receipt. The samples are artificial, donors are cyclic shifts, and carry is independent; no semantic counterfactual claim follows.

A subsequent historical calibration audit identifies large differences between the assumed and actual laws. Actual source RMS is932.47 versus sphere-generated475.96; source energy under the sphere is only26.05%of historical source energy. The actual carry/source RMS ratio1.648 lies within the tested range, so simply widening that ratio is not the obvious repair. Centered source/carry cosine is+.2702, while uncentered cosine is−.1108; omitting means would reverse the apparent relationship. The sum retains90.17%of the sum of separate uncentered energies. Trace-normalized source-covariance relative distance is98.68%, and source-mean discrepancy is82.89%of historical total source RMS norm. Actual input squared radius is1152: normalization-radius matching alone does not match the source law.

These descriptions do not causally isolate which mismatch produced the ranking reversal. They do refute treating normalized isotropic inputs plus an independent carry as a faithful native-source distribution. A new data-free generator would need a different justification; more runs of the same law are not the highest-information step.

The next comparison is finite source-change fitting on historical calibration states, while retaining exact global tensor penalties. For a quadratic q and normalized endpoint inputs x0,x1, q(x1)−q(x0) equals its derivative at (x0+x1)/2 applied to x1−x0. Thus the existing exact response solver can match finite changes without a local RMS linearization. The affine mean/tangent repair makes teacher-minus-student differences equal centered-quadratic differences, so use the common calibration mean explicitly. Five independent direct-evaluation/gradient controls pass below1e-12. This is a data-informed proposal, not a new successful native fit; final normalization and softcap must still be tested afterward.

[Artificial outcomes](SPHERICAL_SOURCE_GEOMETRY_V1.json) · [Historical geometry audit](SOURCE_CARRY_GEOMETRY_AUDIT_V1.json) · [Finite-response controls](FINITE_SOURCE_RESPONSE_CONTROLS_V1.json).
