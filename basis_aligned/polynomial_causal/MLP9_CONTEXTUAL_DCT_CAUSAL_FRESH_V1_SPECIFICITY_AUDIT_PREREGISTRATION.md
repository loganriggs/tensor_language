# MLP9 contextual DCT causal V1 specificity audit

The registered fresh test passed, but its eight isotropic removal controls
matched only the global norm of the packaged response.  That may be too weak
because the target reader can weight token positions and the rank-16 response
subspace anisotropically.

On the now-opened fresh panel, rerun removal with three stronger frozen control
classes, always matching the packaged response norm separately at every token:

1. eight isotropic residual directions;
2. eight random directions constrained to the same rank-16 output subspace;
3. the other three packaged diagonal DCT responses `(0,0)`, `(1,1)`, `(3,3)`.

For each control, aggregate its effect on the frozen target reader across all
64 rows and divide its L2 norm by the packaged `(2,2)` removal effect.  Also
report all-logit norm ratios and cosines.  Threshold `.50` is frozen for every
individual control, matching the original collateral ceiling.

- **A:** exact replay remains within `2e-6`.
- **B:** all eight position-matched isotropic target-effect ratios are ≤`.50`.
- **C:** all eight same-subspace target-effect ratios are ≤`.50`.
- **D:** all three alternate-DCT-pair target-effect ratios are ≤`.50`.

Failure narrows or overturns the original specificity interpretation but does
not invalidate its OOD state prediction, installation, removal, extraction, or
composition measurements.
