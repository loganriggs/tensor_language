# Native conditional MLP11 numerator degree split

CIRCUIT track. Prior exact folded response passes8/8; CPU one-context screen suggests quadratic numerator is small. Test whether this split identifies a dispensable part of the mixed source response at fixed native background. No fit; use all opened-v4 cells and frozen finite-reference source selectors.

Reuse run_v4_mlp_response_fold_v1's native anchors, exact fold and independent local check. Boundary10.8 removes all quadratic numerator coefficients, boundary10.9 removes all linear coefficients. Both retain full residual carry and exact RMS denominator, native additive background MLP, and suffix12–17. Full10.7 is positive control. These are response interventions; not removal of whole original MLP or native source channels.

pred_a_instrument: counts12prefix16native32joined, old native anchors<=1e-8, local fold<=1e-8, readout check<=.001 and exact fold passes every cell.
pred_b_linear_numerator: all8cells meet number error<=.1 and each control<=.05, normalized by weaker singleton number effect.
pred_c_linear_needed: quadratic-only numerator fails at least one cell under identical thresholds. If both pass, do not claim specificity. No bar changes on failure.

Literal deployment price for linear numerator:31104coefficients+31104carry+405norm+1baseline=62614/context; full canonical498070. Native background and suffix, edge writer/amplitude generator and source selectors remain charged. This runner computes dense coefficients before masking as a screen, not an efficient implementation. Twenty explicit MLP11 calls beyond suffixes. No exported dense duplicates. If native omission passes, implement direct reduced compiler and independently verify it, then test held-out amplitudes before claiming reuse.
