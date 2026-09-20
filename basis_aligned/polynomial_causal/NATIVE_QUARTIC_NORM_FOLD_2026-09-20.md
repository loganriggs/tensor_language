# Native quadratic-state fold and two-square polynomial quotient

The exact readout of the quadratic final-state curve passes its instrument checks,
but not the registered all-method prediction gate. Full-radius worst number errors:

| Method | Quadratic fields | Quadratic state, quartic norm | Output cubic |
|---|---:|---:|---:|
| Original5 |8.347%|8.313%|3.416%|
| Full6 |10.257%|10.103%|7.208%|
| Substituted5 |9.569%|9.445%|6.385%|

Full6 still fails10%; do not round it into a pass. All modal errors remain below5%.
Projected first/second derivatives replay1.56e-14, exact state readout4.98e-14,
native/double differences7.44e-6 and previous native outcomes replay exactly.
The error therefore survives correct state derivatives and explicit quartic norms.

State capture uses24first and24second logical JVPs, implemented by nested
reverse-over-reverse autodiff. These are not primitive reverse-call counts.
The validation includes12prefix,112double and88native calls,72local readout checks
and96+96cheap root reverse calls. Total run10.456s versus26.881s for quadratic
fields, both single runs with different derivative procedures; this is not a
repeated performance benchmark. Native generators remain required.

## A useful literature connection, with an executed consequence

Actual search: "site.arxiv.org nonnegative univariate polynomial sum two squares
Gram matrix decomposition". Opened the primary abstract of
[Magron, Safey El Din and Schweighofer](https://arxiv.org/abs/1706.03941).
It states that nonnegative real univariate polynomials admit two-square
representations; the paper studies weighted decompositions for rational inputs.
Our native coefficients are floating point, and we do not import its rational
certification or complexity claims. The following small Gram-boundary algorithm
is derived for our quartic object and checked numerically.

The compiled norm is phi^T G phi with phi=(1,t,t²) and G=R^T R PSD.
Let N have N02=N20=1, N11=-2 and all other entries0. Since
phi0*phi2=phi1², phi^T N phi=0. Thus G+delta*N represents the same polynomial
for every delta. For positive definite G, the PSD interval endpoints follow
from eigenvalues of G^(-1/2)N G^(-1/2). An endpoint has rank at most2.
Factor it, then rotate the two features so only one has a constant term:

    norm(t) = (a+b*t+c*t²)² + (d*t+e*t²)².

RMS epsilon is retained separately. This uses two shared quadratic features and
five coefficients instead of three features and six coefficients. Eight token
numerators stay unchanged; the complete conditional readout goes30->29values
per ray, with one fewer norm square. The cheaper16coefficient cubic baseline
and native generator costs remain on the ledger.

The CPU compiler passed64planted full-rank, deficient and zero cases. All288
saved native readout programs reduced to two squares: coefficient relative
error<=6.18e-16 and output replay<=7.11e-15 over signed radii through±2.
Coefficient equality is the functional identity check; finite-radius checks
also verify execution. No new native effects were fitted or evaluated.

This is a concrete example of polynomial equivalence simplifying a computation
that its original Gram rank obscures. It does **not** identify semantic features,
guarantee a unique basis, extend the two-square claim to multivariate inputs, or
fix the native10.103%prediction failure. Gauge freedom and context-dependent
state/derivative generators remain. The result is a local exact compiler
improvement, not the requested complete circuit.

Artifacts: `source_ood_v2_state_readout_result.json` in the followup directory;
`NATIVE_TWO_SQUARE_NORM_V1_RESULT.json`, `quartic_two_square_quotient.py` and
`audit_native_two_square_norm_v1.py` here.

The next circuit-relevant folding test should use two independent intervention
axes and predict their joint effects. A collection of accurate one-dimensional
rays does not establish cross-component composition.
