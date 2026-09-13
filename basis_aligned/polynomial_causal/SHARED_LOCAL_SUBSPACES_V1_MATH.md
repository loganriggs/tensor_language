# Can optimization find the shared/global and private groups?

13 September 2026, 05:08 UTC. This follows the [full-U feasibility calculation](FULLU_SHARED_LOCAL_FEASIBILITY_V1_MATH.md). It tests the algorithm on known synthetic structure, before native fitting.

**Three of four independent starts recover the planted shared/private function family. One reaches an incorrect stationary solution.** A targeted group replacement fails to escape that solution. Restarts help on this fixture; neither conditional optimality nor a tiny gradient certifies global recovery.

## Computation

For the exact folded-function coordinates \(X\in\mathbb R^{V\times d}\), fit

$$
X_v\approx a_vP+b_vQ_{c(v)},
\qquad
\min\frac{\sum_v\|X_v-a_vP-b_vQ_{c(v)}\|^2}{\|X\|_F^2}.
$$

The global bank \(P\) has \(g\) rows; each private bank \(Q_k\) has \(r\) rows. Individual banks have orthonormal rows. Different banks may overlap, so the model permits a common parent with private corrections without forcing globally disjoint input spaces. The native caller must retain the vocabulary mean separately, as in the feasibility note.

[The implementation](shared_local_subspaces_v1.py) alternates three conditional steps:

1. Subtract the current private contribution. The best rank-\(g\) approximation of the remainder supplies the global bank and its coefficients, using an exact SVD.
2. Subtract that new global contribution. Within each current group, an exact rank-\(r\) SVD supplies its private bank and coefficients. Empty groups retain their old bank.
3. For each row, try every joined bank \([P;Q_k]\). Choose the span giving the smallest residual and solve jointly for both global and private coefficients. An SVD of the small joined bank handles overlap and numerical rank; projection onto an orthonormal span avoids assignment decisions based on ill-conditioned coordinates.

Each step can retain the previous solution within its allowed variables, so exact arithmetic cannot increase the objective. This gives monotonic conditional optimization, **not** a convex joint problem. The group assignments are discrete; a different arrangement may require crossing an intermediate worse fit.

The global conditional SVD can also be expensive at native dimensions. The code currently uses direct SVD, rather than an approximate randomized subspace update. Measure native execution cost before choosing a large iteration budget; use an exact smaller Gram eigensolve where beneficial, with a replay check.

## Recovery experiment and executed counterchecks

The fixture has 320 rows in 20 dimensions, two shared functions, four groups, and two private functions per group. Shared coefficients have twice the private standard deviation. Each true group has 80 rows. The solver sees only the resulting matrix and the requested widths, **not** the planted banks or labels. The stronger common signal makes this an informative but favorable fixture.

| Seed | Initial squared relative error | Final squared relative error | Sweeps | Outcome |
|---|---:|---:|---:|---|
| 0 | 0.06605 | 5.65e-11 | 9 | Function recovered |
| 1 | 0.07258 | 3.28e-11 | 9 | Function recovered |
| 2 | 0.07426 | 0.01699 | 150 | Wrong stationary solution |
| 3 | 0.06116 | 2.88e-11 | 9 | Function recovered |

Recovery here concerns the fitted function values, not uniqueness of a particular shared/private factorization. Equivalent bank coordinate changes alter the predictions by at most 1.84e-15. No meaningful objective increase occurs. The failed start gains nothing from one more complete sweep.

The normalized objective's bank derivative, with coefficients solved optimally and assignments fixed, is

$$
\nabla_P f=\frac{2A^\top(\widehat X-X)}{\|X\|_F^2},
$$

projected onto the tangent space of row-orthonormal \(P\). Private banks have the analogous restricted-row derivative. An independent finite difference after re-solving coefficients agrees to 2.80e-10 relative error. The failed start's largest intrinsic gradient is 4.21e-15: a small gradient can coexist with this nonzero error.

The executed escape test takes the 32 largest-residual rows, builds a candidate private bank from their residual after the global contribution, replaces each existing private bank in turn, and runs 100 conditional sweeps. The best result is 0.01742 squared error, worse than the original 0.01699. Keep the original result; this heuristic did not repair it. Three independent initializations did recover the same target computation, so this failure does not establish absent structure or an unusable representation.

Receipts: [four-start recovery](SHARED_LOCAL_SUBSPACES_V1_CONTROL.json), [group exchange](SHARED_LOCAL_EXCHANGE_V1_CONTROL.json), and [gradient check](SHARED_LOCAL_GRADIENT_V1_CONTROL.json). All executed on CPU; no native token-function fit or text validation has occurred here.

## Native experiment consequence

Use multiple initializations, retain the best objective, and separate four claims: numerical correctness, differential local convergence, consistency across starts, and useful compression. Run both candidate capacities from the feasibility note: 64 shared / 32 groups / 8 private, and 128 shared / 64 groups / 16 private. Charge their group IDs and dense reader banks. Compare against optimal global compression at the same storage budget.

For a native negative result, check whether more starts, group imbalance, overlapping banks, and the coefficient metric itself could explain the miss. Do not infer an arithmetic-circuit lower bound from these function-subspace fits. A useful native fit still needs frozen behavioral validation and explicit input/normalization dependencies before it supports extraction or reuse claims.
