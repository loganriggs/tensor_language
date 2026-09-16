# Subject-number response-weighted selective removal V1

Registered after the fourth-corpus causal result and before removal outcomes are
computed. The fourth-corpus rows are now open; this test concerns a new intervention
trait, not another OOD claim.

Recompute the frozen response-weighted coefficient $\hat\alpha$ from the native
recipient/background state and audit it against the stored fresh-causal receipt.
Let $h_0$ be the reconstructed recipient L11H3 head and $h_1$ the exact
opposite-number head. Evaluate:

- base: $h_0$;
- exact: $h_1$;
- candidate removal: $h_1-\hat\alpha u$;
- symbolic-law removal: $h_1-\alpha_{law}u$; and
- four null removals $h_1-\hat\alpha r_k$, where each $r_k$ is a deterministic
  unit vector orthogonal to $u$.

The null write on every row has exactly the same norm as the candidate write and
is installed at the same head-output site. No removal coefficient is fit.

For target effect $E=h_1-h_0$, define the removal residual as the downstream
effect of the removed arm relative to base. Candidate removal passes if residual
RMS divided by exact-effect RMS is at most `.50`, and the removed behavioral
amount predicts $E$ at cosine `>=.90`, relative L2 `<=.50`, and sign agreement
`>=.90`. Its residual ratio must be at least `.10` below symbolic-law removal and
at least `.20` below the median equal-norm null.

Selectivity is evaluated on five fixed unrelated final-logit contrasts:
work/jobs `(670,3946)`, cat/dog `(3797,3290)`, red/blue `(2266,4171)`,
Monday/Tuesday `(3321,3431)`, and apple/orange `(17180,10912)`. For each, divide
the RMS change caused by candidate removal (candidate-removal arm minus exact arm)
by target-removal RMS. The maximum ratio must be at most `.25`.

Instrumentation requires 512 rows per arm, exact decomposed closure at most `5e-5`,
finite metrics, target exact-effect RMS at least `.02`, equal candidate/null write
norms within `1e-6` relative, and recomputed candidate coefficients matching the
prior receipt within `1e-5` absolute. A pass establishes selective removal for the
extracted write at L11H3; it does not establish removal of every native number path.
