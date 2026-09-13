# Folding shared key/value source dependence

13 September 2026. This extends the existing linear value-map pullback with a
specific algebraic constraint: both key readers and the value read the same source.
It does not yet compress the normalized retained head17 interaction.

For fixed query, let the two effective source readers be vectors
$a,b\in\mathbb R^{1152}$ and the current-value map be
$F\in\mathbb R^{128\times1152}$. For a standard Gaussian source $y$, the
unnormalized attention write is $(a^Ty)(b^Ty)Fy$. Its second moment is

$$
H=F\left[cI+2\|b\|^2aa^T+2\|a\|^2bb^T
+4(a^Tb)(ab^T+ba^T)\right]F^T,
\qquad c=\|a\|^2\|b\|^2+2(a^Tb)^2.
$$

The term $cFF^T$ treats the score product and value covariance independently.
The correction has rank at most two for a fixed query; averaging different queries
can increase its rank. This gives an implicit metric for fitting downstream maps
without materializing a degree-six source tensor. Gaussian contraction is a
text-independent modeling choice, not evidence of in-distribution importance.

The degree-six identity passes exact four-point-per-dimension Gauss–Hermite
quadrature in five dimensions, relative error $1.01\times10^{-15}$.
The actual head17.2 diagnostic uses 256 independent Gaussian queries, shared
Gaussian key/value sources, and zero relative rotary displacement. Query and
source are independent here even though their rotary frames agree. It includes
the learned current-value multiplier, but excludes inherited first-layer values,
all query/key/value normalization and the retained intervention gates.

Compared with independent key/value contraction, covariance changes by 2.62%
in relative Frobenius norm. The correction accounts for 1.03% of total trace.
Both covariances remain rank 128; the discarded trace after rank 64 moves from
33.32% to 33.10%. Thus this broad Gaussian marginal alone gives little evidence
of substantial new head-mode compressibility. It is not a negative result about
the retained interaction, other positions, normalized execution, or more general
graph compression. A useful extension must retain those dependencies rather
than repeat the earlier value-only spectral fit.

Receipts: `SHARED_KEY_VALUE_MOMENT_V1_CONTROL.json` and
`SHARED_KEY_VALUE_WEIGHTS_V1_RESULT.json`. Implementations:
`shared_key_value_moment_v1.py`, `check_shared_key_value_moment_v1.py`, and
`check_shared_key_value_weights_v1.py`.


## 14:55 — Bound the metric change, including weak directions

The earlier2.62%Frobenius covariance difference did not bound what happens in
low-energy directions. The new CPU generalized-eigenvalue check addresses that
specific loophole, using exactly the same256Gaussian queries, seed and scope.
Let $H_0$ be the independent key/value covariance and $H$ the shared covariance.
Whiten with $H_0=CC^\top$ and diagonalize $C^{-1}HC^{-\top}$. Its eigenvalues
range from1.0001037604to1.1070519560. Recovered extremal Rayleigh-quotient
witnesses agree to the recorded floating-point precision.

For every downstream linear error map $E$,

$$
1.0001037604\,\operatorname{tr}(EH_0E^\top)
\leq \operatorname{tr}(EHE^\top)
\leq 1.1070519560\,\operatorname{tr}(EH_0E^\top).
$$

This follows by multiplying the positive-semidefinite matrix inequalities by
$E$ and $E^\top$ and taking the trace. The same statement holds after summing
independent background-coordinate contributions to a mixed operator.
Within any identical feasible compression class, a **global** optimizer for
$H_0$ therefore has at most1.1069371times the optimal$H$squared error, or
1.0521108times its error norm. It is not a guarantee about a locally optimized
candidate. Nor is it a bound comparing either metric with the raw residual/head
coefficient norm: the independent value pullback is already part of$H_0$.

The registered <=1.2distortion prediction passes. This is a narrow reason to
avoid an expensive refit based solely on this already-tested marginal correction.
It does not bound normalized attention, inherited values, retained gates,
correlated background/source inputs or other positions. Those are precisely
what the proposed setting2 producer-aware objective must add. No fit, GPU run,
new behavioral test or absence-of-structure claim follows.

[Calculation](check_shared_key_value_metric_bound_v1.py) ·
[Primary receipt](SHARED_KEY_VALUE_METRIC_BOUND_V1_RESULT.json).
