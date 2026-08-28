# Quadratic MLP composition diagnostic

## Purpose

An isolated replacement compares `g(z_live)` with the native quadratic map
`f(z_live)`. In a composed ship, earlier replacements change the typed interface to
`z_composed`. The new diagnostic measures the resulting change in replacement error

\[
e(z_{composed})-e(z_{live}),\qquad e=f-g.
\]

This is the local quantity that an isolated fidelity score omits.

## Exact identity and certificates

Because `e` is homogeneous quadratic,

\[
e(z_c)-e(z_l)
=J_e\!\left(\frac{z_l+z_c}{2}\right)(z_c-z_l)
\]

exactly, not just to first order. `residual_secant_diagnostics` reports, per row:

- input-state shift norm;
- observed residual-drift norm;
- exact midpoint-JVP reconstruction error;
- the midpoint Jacobian's induced 2-norm and resulting local upper bound;
- the residual tensor output-unfolding bound for the actual pair of state norms.

The expected ordering is

\[
\|e(z_c)-e(z_l)\|_2
\leq \|J_e((z_l+z_c)/2)\|_2\|z_c-z_l\|_2
\leq \|\Delta T_{(out)}\|_2
 (\|z_l\|_2+\|z_c\|_2)\|z_c-z_l\|_2.
\]

Dense randomized CPU oracles verify the secant equality and both inequalities.

## Composed-evaluator integration

For the same frozen token rows, capture MLP input states once with upstream modules
live and once with the current upstream replacement prefix. Decode `f` and `g`, call
`residual_secant_diagnostics` in bounded row batches, and retain paired row records.
The exact-reconstruction error is a wiring invariant: failure means the wrong typed
boundary or factors were supplied. Report drift and bound quantiles alongside CE,
but do not turn them into another fidelity score.

Interpretation:

- small state shift, large local coefficient: the replacement is intrinsically
  sensitive near the reached state;
- large state shift, moderate coefficient: the upstream program is leaving the
  isolated-fit distribution;
- loose global but tight local bound: worst-case tensor sensitivity is irrelevant
  on reached states;
- small diagnostic drift but large composed CE: investigate downstream amplification
  or a different module boundary rather than refitting this MLP.

This diagnostic does not replace held-out, composite, extraction, removal, or OOD
evaluation. It attributes a composition gap after one has been measured.

## Literature mapping

[Oseledets' tensor-train decomposition](https://doi.org/10.1137/090752286) makes
stable approximation and rounding depend on low-rank auxiliary unfoldings and their
SVDs. The relevant implementation lesson here is to certify a contraction using the
operator norm of the appropriate unfolding instead of total Frobenius energy. The
bilin18 MLP tensor is an order-three partially symmetric tensor, not a TT chain, so
the TT quasi-optimality result is not claimed directly.

[De Lathauwer, De Moor, and Vandewalle's multilinear SVD](https://doi.org/10.1137/S0895479896305696)
provides the mode-unfolding framework used for the output spectral norm. The exact
midpoint identity is stronger and architecture-specific: it follows from the
homogeneous quadratic bilinear MLP itself and requires no approximation theorem.
