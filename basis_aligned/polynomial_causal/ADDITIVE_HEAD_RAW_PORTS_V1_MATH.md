# Reconstructing the fourth attention input from three raw-state inputs

13 September2026,03:12UTC. This reuses the nested-RMS identity already developed in DIRECTIONAL_INTERACTION_LOGIT_V1_MATH.md. The new application is the additive-input corner required by the conditional head17.2 mixed-response predictor.

Let r_N,r_C,r_R be preattention residual states of shape T by1152 under native, child-removal and remainder-removal executions. In real arithmetic, the last block's affine residual re-entry preserves the additive relation:

$$
r_A=r_C+r_R-r_N.
$$

All three conditions share the same initial-state term. For any head projection matrix M, linearity gives

$$
p_{M,A}=p_{M,C}+p_{M,R}-p_{M,N},\qquad p_{M,j}=Mr_j.
$$

Define rho_j=mean(r_j squared)+epsilon. A residual RMS followed by projection and head RMS simplifies exactly to

$$
\operatorname{RMSNorm}_{128}\left(M\frac{r_j}{\sqrt{\rho_j}}\right)
=\frac{p_{M,j}}{\sqrt{\operatorname{mean}(p_{M,j}^2)+\epsilon\rho_j}}.
$$

Thus the two rotated normalized QK scores can be computed from the three sets of raw linear projections, the four residual norm arrays, and the actual positional operators. Current values use p_V/sqrt(rho), mixed with the shared first-layer values using the actual signed coefficient. Both QK factors and both normalization stages remain represented.

The fourth residual norm must be supplied or computed from the three raw states:

$$
\rho_A=\frac{\|r_C+r_R-r_N\|^2}{1152}+\epsilon.
$$

The three individual norms alone do not supply the cross inner products. Nor do the head's raw projections determine this norm: they leave a nontrivial residual nullspace. A planted actual-weight control adds a vector in the joint nullspace of all five Q/K/V maps. Projection change is8.96e-15 relative, yet normalized mixed values change68.4%. Omitting the norm would silently discard real input dependence.

## Executed algebra and native test

The CPU implementation uses actual head17.2 weights and actual rounded rotary tables. On three synthetic13-token contexts, reconstructing all three attention ports from the additive projections agrees with an independent residual-RMS, projection, head-RMS and rotation computation within2.20e-15. No fitting is involved. This is algebra validation, not native finite-precision validation.

The registered native comparison uses120 previously scored regional prefixes. It reconstructs all four score/value corners from only the N/C/R raw preattention states plus first-layer values. The A norm is computed from their sum; no A scores or values are supplied to the candidate. It tests full mixed-write effect within2% of native head effect and compact three-group effect within2% of the prior supplied-port version, separately in five groups. Native reference trajectories and additive final background remain available for scoring.

The test still runs600 full forwards and720readouts because it collects the validation references. It doesnot yet demonstrate an actual reduction in deployed fullforward count. The current interface also still receives three1152-wide raw state arrays; changing this to only projected features requires retaining the missing norm information. A closed input interface is progress toward extraction, not an autonomous token-to-logit model.

[Algebra and nullspace control](ADDITIVE_HEAD_RAW_PORTS_V1_CONTROL.json) · [Executor](additive_head_raw_ports_v1.py).
