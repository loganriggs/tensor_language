# Native MLP polarization depth profile — descriptive preregistration

**Frozen before computing MLP3--17 spectra: 2026-08-30 00:07 UTC.**

The shipped table program has a causal extra-rank knee between MLP9 and MLP10.  This
weight-only analysis asks whether that boundary is already visible in the exact native
quadratic coefficients, or instead arises from the residual states and suffixes that
use those coefficients.

For every native MLP site `0:18`, fix the second polarization input to the checkpoint
coordinate vector $e_0$ and form

$$
A_{e_0}=\frac12D[\operatorname{diag}(Re_0)L+\operatorname{diag}(Le_0)R].
$$

Compute all singular values in float64 and report the conservative numerical-rank
lower bound, $\sigma_{513}$, $\sigma_{769}$, and the optimal relative Frobenius tails
after ranks 512 and 768.  MLP0--2 must exactly reproduce the existing certificate.

The prospective descriptive rule is:

- call an adjacent coefficient knee only if MLP10's relative rank-768 tail is at
  least 1.20 times MLP9's;
- call a group coefficient shift only if the median relative rank-768 tail over
  MLP10--17 is at least 1.20 times the median over MLP0--9;
- otherwise classify the shipped knee as not explained by this coefficient slice.

This is mechanism-generating, not an independent confirmation of the already-known
shipped-program knee.  It makes no natural-text CE, reachable-state, semantic,
consumer, extraction, edit, or OOD claim.
