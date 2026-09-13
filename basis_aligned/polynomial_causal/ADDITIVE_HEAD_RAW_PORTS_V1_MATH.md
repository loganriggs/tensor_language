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

## 03:17 — Native reconstruction passes; projected-input norm closure derived

The native validation completed600forwards and720readouts in8.58seconds. A/B/C allpass. All prior native and four readout anchors replay exactly. Full reconstructed mixed-effect errors are0.075%,0.134%,0.042%,0.050%,0.052% across theoldanchor andfournewercontext groups. Compact reconstructed effects differ fromtheprior supplied-port compact predictor by0.056%,0.170%,0.072%,0.054%,0.055%. Maximum full-effect score-margin error is1.91e-6.

An independent per-prefix outcome audit retains all119 nonzero reference signs and the remaining exactzero for both reconstruction comparisons. Against the actual fullhead mixedwrite, the reconstructed compact predictor remains approximate:4.68% onoldanchors and5.19%,3.95%,3.64%,3.52% onthefournewer groups. All95nonzero signs andonezero match onthe96newergroup examples. These rows were previously scored; this is interface validation, not newheldout evidence.

This removes the need to supply the A-corner head scores/values. It doesnot yet prove fewer executed native forwards: the test still computed P/A references. A three-trajectory deployment must be run separately, using onlyN/C/R andtheirfinalbackground states.

The next executed CPU step makes the missing norm information explicit. Put deltaC=rC-rN anddeltaR=rR-rN. Then

$$
\rho_A=\rho_C+\rho_R-\rho_N+
2\operatorname{mean}(\delta_C\odot\delta_R).
$$

Thus one cross-inner-product scalar perposition, beyondthethree individualnorms, suffices. `three_corner_head_interface_v1.execute` receives three sets of rawQ1/K1/Q2/K2/V projections, three normarrays, that crossscalar, andshared firstvalues. It computes the fourthnorm andallfour attentionportcorners internally, returning either thefullmixed128-vector orthecompactapproximation. It doesnotinfer thecrossscalar fromprojections. The residual-nullspacecounterexample remains relevant.

Actual-weight synthetic checks comparewithdirectrawA calculation: normidentity2.80e-16, fullmixedwrite3.81e-14, compactmixedwrite2.81e-14. The projected-only wrapper hasalgebra validation; thenative experimentabove suppliedrawstates andcomputedtheA normdirectly. Do not conflate these distinct interface receipts.

The current wrapper carries allfive128-dimensional projections pertoken percorner; onlythefinalquery projections are mathematically needed, butthat additional interface optimization isnotimplemented. Projection/norm/crossscalar generators andnativebackground/readout remainoutside it. No fullmodel simplicity orautonomousinputgeneration claim.

[Native result](ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_RESULT.json) · [Per-prefix audit](ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_AUDIT.json) · [Projected interface control](THREE_CORNER_HEAD_INTERFACE_V1_CONTROL.json) · [Projected interface](three_corner_head_interface_v1.py).
