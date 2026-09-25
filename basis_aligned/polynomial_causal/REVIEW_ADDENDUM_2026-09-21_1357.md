# Concurrent-review addendum: conditional-core conditioning

Recorded 2026-09-21T13:57:44.295083+00:00. The initial review check found 10:50 UTC newest; immediately before publication, the substantive [13:50 review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_1350.md) had appeared. It is current. This addendum does **not** reset its clock; the nominal next deadline remains **16:50 UTC**. No new three-hour review, experiment, code repair, agent, GPU job, queue change, or primary-receipt edit was performed. The concurrent-review stop rule supersedes the planned CPU consequence.

## LITERATURE_SEARCH

Actual queries executed:
- `Cai Li 2021 identification matrix joint block diagonalization perturbation AISTATS`
- `Golub Pereyra 2003 variable projection separable nonlinear least squares rank deficient`

Opened primary sources:
- [Cai and Li, Identification of Matrix Joint Block Diagonalization](https://proceedings.mlr.press/v130/cai21a/cai21a.pdf): accessible PDF. Local quadratic pairs fit the full-column-rank mixing/block-core model on their support; the overlapping global graph does not automatically satisfy its identification assumptions. This agrees with the completed review and does not change its plan.
- [O'Leary and Rust, Variable Projection for Nonlinear Least Squares Problems](https://www.cs.umd.edu/users/oleary/software/varpro.pdf): accessible author-hosted PDF. Eliminating linear coefficients is relevant to our exact conditional-core solves; it does not guarantee global recovery of nonlinear input spaces.
- [Golub and Pereyra, differentiation of pseudoinverses](https://epubs.siam.org/doi/abs/10.1137/0710036?journalCode=sjnaam): search excerpt states a constant-rank hypothesis; opening the publisher page returned **Internal Error**. Full text was not verified. No skipped or failed access is counted as a completed reading.

## Additional mathematical observation

Read `direct_tensor_match/orthogonal_private_varpro.py`. For orthonormal shared/private input bases P,D, its conditional symmetric core satisfies

$$C-JCJ=R,\qquad J=D^T PP^T D.$$

If the principal angles between the spaces are theta_i, eigenvalues of J are cos²(theta_i). In its eigenbasis the linear solve divides element (i,j) by

$$1-\cos^2(\theta_i)\cos^2(\theta_j).$$

This is a precise link between variable-projection conditioning and shared/private overlap. For lambda_max(J)<1 the Frobenius inverse-operator norm is exactly 1/(1-lambda_max(J)²). At an exact intersection the solve loses invertibility; a pseudoinverse convention would define a different handling of that stratum. A valid gradient at a regular point is not a uniform guarantee across that boundary. A ridge changes the objective and must retain the original reconstruction score. This derivation supports logging principal-angle/core-solve conditioning beside convergence in existing runners; it neither requires another framework nor changes the scheduled review's next cost-allocation test. No new numerical validation is claimed.

## Read-only receipt and organization reconciliation

The completed four-arm `PENCIL_JOINT_REFIT_V1.json` agrees with the new review: inherited/algebraic covariance errors 7.8931%/8.1935%; native-isotropic fits 49.7619%/50.0072%. Instruments pass, original fidelity gates fail. Receipt SHA256 `fbb5ba8cec620074c46030ec1c53455f046e502bbdada5644736267ea92978b2`. These are fit/opened diagnostics, not fresh causal confirmation. All four compiled graphs report 1,058,124 physically stored floats, 4,224 index integers, 1,056 source products and 1,047,648 source multiplications; native ports and common downstream computation remain external.

Read the circuit registry, computation-path registry, module dossier slices, graph registry, explanation index, hourly 12:55 receipt, and continuation-group dossier. Their scopes differ: the generated graph registry's two four-trait packages are not certification of the current pairwise source graph. The continuation dossier retains its behavioral evidence and native-input dependencies. Its latest viewed entry was 11:32, while the board and primary receipts were already at the native pencil refit. The explanation index still advertised 11:32 as latest follow-up, and the direct-study README's opening direction was 00:18. Treat these as navigation lag; do not overwrite concurrent publications. Startup systemd language is historical; read-only Supervisor status confirmed bqrunner/bqrunner2 RUNNING. No service action was taken. Focused refactoring is deferred because current shared compiler/assessment helpers already exist and the concurrent review has assumed the next scientific action.

## BASELINE_COMPARISON and evidence limits

The baseline authority is `DECOMPOSITION_BASELINES_2026-09-20.md`. The pair baseline has 1,340,940 floats, 3,456 indices, 1,152 products and 1,330,560 source multiplications. Compare identical six outputs, z/h ports, precision, pair-normalized metric and downstream errors. This local saving is not a whole-model saving. Native factors remain fidelity references. Conventional spectral/Tucker/HT results at other scopes cannot fill missing same-output, matched-total-cost controls for this new graph. No new baseline run was performed.

Five-property status for the current cheaper pairwise graph: **simple:** measured local arithmetic/storage saving, complete dependency price absent; **held-out/OOD prediction:** original component fidelity fails, no new fresh success; **extracted:** compiled conditional z/h interface, upstream closure absent; **selective:** not established for this new candidate against appropriate controls; **composes/reuses:** projections are shared algebraically, behavioral composition and stable feature identity unproven. Prior continuation evidence does not transfer automatically. No new distinct context cells were evaluated.

**REDTEAM_POSITIVE:** planted two-stage recovery is not native discovery; beam-width dependence, wrong-space low-error solutions, unchanged nonlinear product counts, and free native ports limit that success. **REDTEAM_NEGATIVE:** the native miss is not a representation impossibility theorem; exact/full-capacity controls, corrected contiguous gradients, and improved planted recovery under equivalent coordinates preserve optimizer and conditioning alternatives. The additional conditioning derivation above sharpens that distinction without modifying the live sweep.

Handoffs remain those of the completed review: bounded weights-only private-width allocation within the original price/gates; later circuit work must freeze any faithful candidate before fresh OOD/selective/composition checks, with forward response census and matched nulls. No data-based circuit screen during the active user focus.
