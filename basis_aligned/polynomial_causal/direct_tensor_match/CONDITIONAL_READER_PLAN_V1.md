# Does the learned input dictionary discard necessary information?

2026-09-20 21:46 UTC. The26/12/10-product programs all fail the code-mode1
removal test. Before enlarging a core or changing the optimizer, estimate an
input-information floor under two explicit Gaussian laws.

For x=mu+Lz, z~N(0,I), let Q span the row space of readers*L.
Generate x,x' with shared Q^Tz and independent orthogonal Gaussian components.
Then for the native polynomial F,

$$
\tfrac12\mathbb E\|F(x)-F(x')\|^2
=\mathbb E\operatorname{Var}(F(x)\mid Q^Tz).
$$

Every deterministic program using only those readers has at least this MSE.
This is an expectation identity; the finite Monte Carlo estimate is not a
certified lower bound. Report uncertainty and numerator/denominator separately.
The CPU quadrature oracle uses F=z0²+z1²+z0*z2, observing z0: exact MSE floor3,
target energy9. It checks the pair estimator against the known conditional mean.

Two fixed reader dictionaries: original26-product union of U,V (<=32 linear
directions), frozen10-product union of A,B (<=12). Two laws: calibration mean
and empirical centered covariance; zero-mean identity covariance. These are
weights-first artificial probes; they do not establish actual-text/OOD floors.
For each law/dictionary use2048 independent pairs in batches64, seeds260922+arm.
Evaluate both the native full reduced output and four fixed canonical scalars.
Report actual student MSE under the same law for comparison, without fitting.

Registered predictions:
- pred_a_instrument: reader coordinates and frozen student outputs agree across
  paired inputs to relative1e-5; CPU oracle error<1e-12.
- pred_b_original: calibration Gaussian original-reader normalized floor<.10.
- pred_c_compressed: calibration Gaussian compressed-reader floor<.15.
These are hypotheses that the dictionaries retain enough information, not
assumptions. Isotropic results reported regardless of outcome. Also report
mode1 floor and block-resampled95% intervals. Native polynomial scale and
fixed output frame follow prior fits; no normalization is silently polynomialized.

No parameter changes, semantic labels or circuit promotion. If floors are
large, new readers are required under that law; if small, this does not prove
that optimization is the only obstacle (core architecture and measure matter).
