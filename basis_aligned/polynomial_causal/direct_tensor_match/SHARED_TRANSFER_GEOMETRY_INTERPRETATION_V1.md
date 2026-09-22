Calibration-transfer geometry diagnostic

22 September2026,00:06UTC. Fixed root1 interface, same512/768root-product dictionaries, two starts. CPU2.59seconds. Diagnostic only; no candidate selection/export.

Hypothesis tested: poor transfer might chiefly arise from a few weakly constrained calibration directions or high-leverage evaluation states. The measured result does not support that simple explanation. Removing weak singular directions does not repair the failure, and high-leverage rows do not dominate weighted error.

The weighted least-squares evaluation oracle gives the exact finite-design decomposition

$$
\|\Phi_e c_c-y_e\|_{W_e}^2
=\|\Phi_e c_e-y_e\|_{W_e}^2
+\|\Phi_e(c_c-c_e)\|_{W_e}^2.
$$

Here c_c is the calibration-trained readout and c_e the evaluation-fitted oracle. Orthogonality holds because c_e minimizes the weighted evaluation least-squares objective in the same dictionary. This does not make c_e deployable or identify why c_c differs.

| Dictionary/start | Calibration-fit evaluation sensitivity error | Evaluation oracle error | Excess error above oracle (quadrature term) | Top10% leverage rows: fraction of weighted squared error |
|---|---:|---:|---:|---:|
|512/1101|20.34%|9.70%|17.88%|8.84%|
|512/1102|21.63%|10.79%|18.74%|10.47%|
|768/1101|20.77%|7.85%|19.23%|7.04%|
|768/1102|22.21%|8.68%|20.45%|9.52%|

Errors in the third and fourth columns combine as squares, not arithmetic differences. About75–86% of calibration-fit squared evaluation error is excess above the finite-dictionary optimum. Enlarging the dictionary improves oracle capacity while increasing this excess term.

Leverage is phi(x)^T(A^T A)^-1 phi(x), up to a common calibration-row factor, where A is the column-scaled sensitivity-weighted calibration design. Evaluation-row sensitivity is not multiplied into leverage; it is used in the reported residual energies. The top10% leverage group carries roughly9–10% of target energy and7–10.5% of squared error. Thus error is not concentrated in that particular extreme group. This is not proof that distribution shift is absent.

All predeclared relative singular cutoffs0,.001,.003,.01,.03 are retained in the JSON. Cutoff.003 gives a modest20.34→19.90% improvement for512/start1101, essentially none for512/start1102, and worsens both768 fits. Larger truncations worsen all fits, up to~25%. We do not select a winning cutoff on this opened evaluation panel. Weak-mode truncation is not a demonstrated repair.

Redteam: an exact linear target under shifted evaluation inputs is recovered to numerical precision; the Pythagorean identity replays below5e-16 relative target energy on native dictionaries. These controls catch solve/decomposition errors but do not validate semantics or explain the statistical gap. Underlying feature producers already used calibration information; these are conditional diagnostics, not an independent training study.

Decision: retain the queued Gaussian-guided structural edits unchanged. More readout conditioning sweeps have low priority. If the fixed-producer search also fails, continuous learning of shared quadratic producers under the mixed functional objective is the stronger comparison to the successful larger mixed-CP fit. No claim of semantic discovery follows from either route without further native/OOD tests.

Artifacts: [results](SHARED_TRANSFER_GEOMETRY_V1.json), [implementation](audit_shared_transfer_geometry.py), [pool-capacity comparison](SHARED_POOL_RESPONSE_CAPACITY_INTERPRETATION_V1.md).
