# Output-specific low-degree corrections help modestly but fail the screen

22 September 2026, 02:16 UTC. The 24-second CPU screen tested a different structural response to failed output balancing: retain the CP512 parents and add separate weight-derived corrections only to fixed output coordinates4–15. Coordinates0–3 are exactly unchanged at the selected polynomial interface. This does not protect full-model behavior after nonlinear normalization.

Under the calibration Gaussian x=mu+Sz, compute native-minus-parent Hermite coefficients m,l,Q directly from weights. A rank-r correction for one output is

$$
c(z)=m+l^\top z+\sum_{j=1}^{r}\lambda_j[(v_j^\top z)^2-1].
$$

The signed eigenpairs of the symmetric Q are ordered by eigenvalue magnitude. Rank0 retains only mean and linear terms. Full rank1152 retains the complete degree0/1/2 Hermite projection. This projection is optimal within that degree class under the specified Gaussian, not necessarily on text. The coefficient spectrum is not a lower bound on the cost of an arbitrary arithmetic circuit. The correction is nonhomogeneous and does not preserve native global evenness; this limitation is unchanged from other local data-informed approximations.

Both CP parents and ranks0/8/16/32/1152 were fixed in advance. No text labels were fitted and no evaluated rank was selected as a new candidate. The primary rank16 condition required at least15% relative improvement in both small-output value RMS and same-token response RMS for both seeds on the original opened panel. It **failed**.

| Rank per output | Added square products | Added coefficients in compiled eigenprogram | Original value error ratio, seeds1001/1002 | Original response error ratio, seeds1001/1002 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 13,836 | .994 / .994 | .997 / .998 |
| 8 | 96 | 124,620 | .953 / .947 | .984 / .979 |
| 16, primary | 192 | 235,404 | .927 / .917 | .949 / .957 |
| 32 | 384 | 456,972 | .883 / .874 | .913 / .928 |
| 1152, diagnostic | 13,824 | 15,966,732 | .666 / .665 | .819 / .832 |

Ratios below one improve over the frozen CP parent. These ratios use RMS of per-coordinate relative errors, not pooled output energy. Rank16 yields about49–50% small-feature value error and56–59% response error on the original panel. Its directions can be compiled into original-coordinate affine readers, without a dense whitening operation at runtime. The listed storage includes those readers, their affine offsets, signed eigenweights and output biases. It is one explicit implementation price; dense or shared alternatives may price differently.

On the larger opened256-document panel, rank16 leaves **52.72/54.10%** small-feature value RMS and **54.37/55.80%** response RMS on the2,494 matched pairs. Full quadratic leaves40.15/42.13% value RMS and41.09/43.14% response RMS. Thus the local quadratic residual contains useful information, but the inexpensive truncation is insufficient, and the full projection remains a poor component replacement despite its large cost. This is not proof that no compact higher-degree/shared correction exists.

Controls: five structurally different affine-CP families agree with independent Gaussian quadrature below4.3e-14 for mean, linear and quadratic projections. Native rank16 compiled-coordinate replay is below7.7e-15. Dominant coordinates0–3 are bitwise unchanged by the correction assembly. The source CP exports and queued native helpers were not modified. The native teacher projection reuses the established checked contraction routine and calibration covariance cache; this audit is not an independent rederivation of that native routine.

Decision: do not promote these corrections or continue an undirected rank sweep. Preserve the evidence that output-local changes avoid the dominant-feature tradeoff but require substantially richer missing computations. The already-queued full/lean conditional native-removal tests remain the next adoption-relevant evidence. A subsequent structural experiment should change the available higher-degree products or the response metric, not assume that reweighting or a few quadratic residual terms solve the problem.

[Preregistered screen](OUTPUT_SPECIFIC_CORRECTION_PLAN_V1.md) · [Projection helper and independent controls](output_specific_correction.py) · [Native audit](audit_output_specific_correction.py) · [All numerical results](OUTPUT_SPECIFIC_CORRECTION_V1.json).
