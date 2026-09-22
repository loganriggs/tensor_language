# A structural obstruction for one shared orthogonal square basis

22 September 2026, 02:51 UTC. This is a bounded mathematical baseline for the true native single-layer quadratic tensor. It does not replace the scheduled three-hour review clock (next04:59), and it is separate from the two-layer quartic approximation experiments.

## Exact object

For the last MLP, with normalized input z in R^1152, define16 fixed linear output readers r_v and

$$
F_v(z)=r_v^\top D_{17}[(L_{17}z)\odot(R_{17}z)]
      =z^\top T_vz,
\qquad
T_v=\operatorname{sym}\!\left(L_{17}^\top\operatorname{diag}(r_v^\top D_{17})R_{17}\right).
$$

Each T_v is1152×1152. This retains every input coordinate but only16 selected output projections; it is not the whole vocabulary-output tensor. The readers are the existing fixed directions, used here to inspect the single native MLP rather than its earlier quartic composition. Bias and normalization remain outside the quadratic part.

The restricted candidate is

$$
F_v(z)=\sum_{j=1}^{1152} c_{vj}(q_j^\top z)^2,\qquad Q^\top Q=I.
$$

It computes one common orthogonal input basis and reuses its squares across every output. Real symmetric matrices admit such simultaneous orthogonal diagonalization exactly when they commute pairwise. Approximate joint diagonalization and commutator size are related in the primary literature; our numerical bound below is derived explicitly rather than importing a theorem with a different norm. [Glashoff and Bronstein](https://arxiv.org/abs/1305.2135).

This is **orthogonal factors plus a diagonal interaction core**. Ordinary Tucker with a full core is not subject to this obstruction: cross terms between different input features remain available. Nor does this test rule out nonorthogonal input dictionaries, overcomplete square features, bilinear products of different forms, block terms or general arithmetic DAGs.

## Pairwise numerical lower bound

Normalize a selected pair independently to ||A||F=||B||F=1. Let commuting approximants have errors E_A and E_B, and let e²=||E_A||F²+||E_B||F². Using commutator norm inequalities gives

$$
\|AB-BA\|_F
\le 2\|B\|_{op}\|E_A\|_F
 +2\|A\|_{op}\|E_B\|_F
 +2\|E_A\|_F\|E_B\|_F
\le 2se+e^2,
$$

where s²=||A||op²+||B||op². Therefore, with c=||AB-BA||F,

$$
\frac{e}{\sqrt2}\ge
\frac{c}{\sqrt2(\sqrt{s^2+c}+s)}.
$$

The denominator sqrt2 is the original pair's joint Frobenius norm. The calculation uses floating-point eigendecompositions, not interval arithmetic, so these are numerical evaluations of an analytic lower bound. They are not confidence intervals or text-state prediction bounds.

| Fixed output pair | Relative pair-error lower bound | Simple constructive upper bound |
| --- | ---: | ---: |
| 0,1 | 5.44% | 41.84% |
| 2,3 | 6.95% | 45.44% |
| 4,5 | 6.60% | 68.61% |
| 6,7 | 6.53% | 67.85% |
| 8,9 | 5.98% | 68.28% |
| 10,11 | 5.70% | 69.51% |
| 12,13 | 5.73% | 68.27% |
| 14,15 | 6.30% | 69.80% |

The upper bound uses either matrix's eigenbasis and drops off-diagonal terms of both, keeping the better result. It is not an optimized joint diagonalization and should not be mistaken for the best achievable error. The broad gap between lower and upper bounds is unresolved.

Every tested pair has a nonzero commutator well above numerical replay scale. Thus one exact common orthogonal square basis is excluded for these selected outputs and consequently cannot exactly represent the full output family either. The percentage bounds refer only to the separately normalized pairs; they cannot be quoted as a lower bound for the differently weighted full output tensor.

## Controls and implications

Five controls cover commuting matrices, nearly commuting matrices, generic symmetric forms, block structure and an exact shared **nonorthogonal** square representation. The nonorthogonal example has nonzero commutator even though its general shared-square representation is exact, demonstrating why the qualification matters. In every control the lower bound is below the error of the constructed commuting approximations. Native dense quadratic contractions replay the original bilinear computation on random probes to the reported numerical tolerance (<1e-10).

This test avoids attributing a structural impossibility to poor optimization. It does not explain every earlier Tucker or HT failure. It says that this particular cheap diagonal-core hypothesis needs relaxation: allow cross-feature interactions, different local bases, nonorthogonal features or additional directions. It offers no semantic identification, native intervention or OOD result.

[Protocol](JOINT_SQUARE_BASIS_PLAN_V1.md) · [Executable derivation checks and native contractions](audit_joint_square_basis.py) · [Results](JOINT_SQUARE_BASIS_V1.json). Earlier [pair-pencil work](PENCIL_SHARED_DISCOVERY_NOTE_V1.md) treats a different, more general congruence/block structure and is not superseded.
