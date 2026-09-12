# A closed feature executor with normalization and residual re-entry

12 September 2026. [Code](closed_feature_program_v1.py),
[registered control](CLOSED_FEATURE_PROGRAM_V1_PREREGISTRATION.md),
[executed result](CLOSED_FEATURE_PROGRAM_V1_RESULT.json).

The recursive native-update subset failed because retained computations needed
omitted inputs. The [norm-aware feature test](NORM_AWARE_FEATURE_SUFFICIENCY_V1_MATH.md)
also showed why reader prediction alone is insufficient: the next norm must be
generated from the proposed state. This executor supplies a positive control
where those dependencies really do close. It is not a native-model result.

## Restricted but genuinely closed model class

Let $B\in\mathbb R^{d\times r}$ have orthonormal columns. Assume every layer's
input readers, output writes and bias lie in this **common** subspace. Write

$$
h_j=\lambda_j h_{j-1}+\mu_j x_0+
B\left[D_j\left((L_j u_j)\odot(R_j u_j)\right)+b_j\right],
$$

where the bracketed expression is in $\mathbb R^r$, and $u_j$ is the normalized
feature-coordinate input after residual re-entry. The physical input may have
a large component outside the feature subspace. Its norm is not discarded.

Encode the original input once:

$$
z_0=B^\top x_0,\qquad v_0=(I-BB^\top)x_0,\qquad t_\perp=\|v_0\|^2.
$$

Since all new writes lie in the feature subspace, the hidden state always has
the exact form

$$
h_j=Bz_j+\alpha_jv_0.
$$

Starting with $z=z_0,\alpha=1$, residual re-entry updates

$$
\widetilde z=\lambda_jz+\mu_jz_0,\qquad
\widetilde\alpha=\lambda_j\alpha+\mu_j,
\qquad
\rho_j^2=\frac{\|\widetilde z\|^2+\widetilde\alpha^2t_\perp}{d}+\epsilon.
$$

Then $u_j=\widetilde z/\rho_j$, the bilinear feature update is evaluated in
$r$ dimensions, and $\alpha_j=\widetilde\alpha$. No original intermediate
hidden vector or native-generated norm is supplied. The ambient dimension $d$
still sets the RMS scale; replacing it by $r$ would change the model.

For an arbitrary linear output map $U$, the final unnormalized readout is

$$
Uh=(UB)z+\alpha Uv_0.
$$

The executor retains the initial complement readout $Uv_0$, and divides by the
exact final norm before the native-style $30\tanh(\cdot/30)$ cap. Keeping this
initial readout is a charged dependency. When $U$ has many rows it can be costly;
it is not an eight-scalar total state just because $r=8$.

## Executed multi-layer and intervention controls

A fixed seeded example has $d=64$, $r=8$, 18 layers, 12 product factors per layer,
and 96 arbitrary output readers. For each of input scales 0.3, 1 and 3, test 64
independent inputs with four arms: baseline, removal of factor(5,1), removal of
factor(12,2), and both removals. The dense reference and reduced executor use
the same weights, re-entry coefficients and float32 epsilon in float64 arithmetic.

All registered predictions pass:

- Full-logit relative error is at most $7.02\times10^{-16}$; feature and squared-
  norm errors are below $8.49\times10^{-16}$.
- Single and joint signed-effect errors are at most $1.86\times10^{-12}$, with
  nonzero effect norms. The executor propagates the edits through later layers.
- Nonlinear joint-interaction norms range from $6.41\times10^{-5}$ to 0.00543;
  relative interaction replay error is at most $8.24\times10^{-10}$. This checks
  interaction prediction rather than assuming additive edits.
- Deliberately omitting the complement norm gives relative logit errors of
  1.33–2.13. The positive result is not due to an unused normalization dependency.

These are synthetic scale-shift and intervention controls. They do not count as
linguistic OOD, selective native manipulation, or a discovered circuit in bilin18.

## Literal price and what remains to discover

The dense control has 48,804 stored scalars. The reduced implementation has
12,788, including its feature frame, full output map for initial encoding, and
feature output map. Each input retains 105 encoded scalars: eight initial
features, one complement norm and 96 complement-readout values. The changing
feature state has eight entries; initial cache storage has not been hidden.

The substantial assumption is common reader/writer support. The native model
has not been shown to satisfy it, and the earlier four-feature counterexample
prevents using those four readers as an exact native solution. A learned
candidate would need a joint producer/readout objective and must preserve this
execution interface or explicitly supply and price additional state. Rank or
conditional reconstruction alone would not establish that condition.

The practical advance is a reusable positive extraction and composition test:
future candidate programs can be evaluated with their own state, their own norms,
and actual combined edits. A model that only works when original hidden states
are inserted will fail this stronger comparison rather than being called closed.
