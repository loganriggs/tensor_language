# Spectral blocks converge reproducibly but do not separate

12 September2026. [Registration](NORMALIZED_COMMUTANT_NATIVE_V1_PREREGISTRATION.md),
[receipt](NORMALIZED_COMMUTANT_NATIVE_V1_RESULT.json). Completed04:59:41 UTC,
30.95seconds measured execution.

Numerical prediction A and cross-seed stability C pass. Separation B fails.
Both starts converge in62operator actions plus2verification actions, with
relaxed eigenvalue estimates0.4178190283 and0.5280104881. Between-start agreement
is4.0e-15; eigen-residuals are at most5.7e-11. The mass matrix is supported:
$K$condition number259.09, and transformed identity removal error2.36e-17.

Rounding the leading witness produces the same partition up to exchanging sides
(overlap1.0), but exact normalized cut is0.900116 versus the0.1bar. Cross-block
energy is44.93% and incident-energy shares are48.01%/51.99%. The previous local
optimizer found lower cuts0.839/0.840 but was unconverged. Spectral reproducibility
therefore establishes a stable numerical object, not independent computation.

The relaxed estimates are not certified lower bounds: independent converged
Krylov runs and small residuals do not prove no smaller eigenvalue exists.
Nevertheless, the registered numerical test does not find nearly independent
blocks, and its rounding remains strongly coupled. No circuit or impossibility
claim follows. Changing the partition threshold after observing this result
would be a different experiment.

The known strict planted local minimum and the successful spectral planted
recovery address one optimization explanation for the preceding negative.
The separate exact shared-parent star shows a broader assumption failure:
cheap reusable arithmetic can have no independent blocks at all. Further
half-block sweeps are therefore demoted. The
[producer-folded shared-parent experiment](COMPOSED_SHARED_PARENT_V1_PREREGISTRATION.md)
is now submitted after interpreting this result, reusing the existing exact
optimizer and unrestricted partners. It tests deliberate reuse across
interacting branches rather than requiring their independence.
