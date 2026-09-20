# Next discriminating control: the fixed linear branch

The full-quartic Gaussian quadratic fit failed its registered improvement bars.
Before more quadratic sweeps, test the linear component held fixed throughout.
For fixed covariance M and x = mu + delta, the Gaussian linear coefficient is

$$B=\mathbb E[\nabla F(x)]=\nabla_\mu\mathbb E[F(x)].$$

The existing rank8 linear reader A defines t=A delta. The optimal writer in
this fixed reader space is W = B M A^T (A M A^T)^{-1}. Eight directional
mean derivatives therefore suffice; no full Jacobian is necessary. The
implemented gaussian_linear_projection.py checks this derivative against an
independent dense quartic oracle. Native execution remains pending.

Profile native JVP memory/runtime before fitting. Then replace only the writer
of each frozen selected quadratic candidate, preserving its readers, quadratic
branch, Gaussian constant, 34560 coefficients and four products. Compare both
metrics, and record the Gaussian linear gain separately from empirical errors.
Predictions: (a) native JVP finite and directional finite-difference agreement
<1e-3; (b) centered candidate panel2 error improves by >=.005 absolute; (c)
export preserves price and replays <1e-5. Null: linear correction also fails;
then degree truncation or the Gaussian metric, rather than one missing
low-degree coefficient, is the likely limitation. No empirical writer fitting.
