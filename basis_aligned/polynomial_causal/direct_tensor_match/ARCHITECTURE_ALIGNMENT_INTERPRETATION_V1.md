# Architecture-controlled alignment and quadratic interactions

21 September 2026, 14:20 UTC. All registered architecture-null comparisons pass, but the magnitude of excess is much smaller than the earlier independent-orientation comparison suggested.

| Geometry | Native | Common coefficient permutation mean | Pair-specific permutation mean |
|---|---:|---:|---:|
| Covariance-shaped | .971312 | .966251 | .948744 |
| Original coordinates | .883601 | .880605 | .802923 |

The common permutation preserves the complete six-read coefficient Gram matrix and trained upstream L/R. Pair-specific permutations preserve each two-read Gram but alter relationships across pairs. Neither preserves each resulting quadratic spectrum. Native exceeds all16 samples in each family/geometry; Monte Carlo upper-tail rank is1/17, not a claim of conventional statistical significance. The original-coordinate native value only exceeds the common-null maximum by .000196. Do not infer semantic circuits from these comparisons.

The next CPU audit derives midpoint principal64 shared spaces separately for each edge. For an orthonormal input frame P, each symmetric quadratic form decomposes orthogonally into the within-P block, its cross block to the full complement, and the remaining block. Frobenius squared energies sum to the original energy; numerical replay is checked below1e-10.

In covariance geometry the within-shared block captures86.98–95.99% of pair coefficient energy, cross terms2.30–9.15%. In original coordinates it captures13.05–20.89%, cross terms17.46–22.24%, with61.65–65.21% outside. Edges overlap: these percentages cannot be added across edges. These are squared-energy fractions, not relative-error percentages or downstream prediction scores.

Interpretation: activation-informed geometry emphasizes a small common input space, but its apparent closure is metric dependent. The geometry does not identify a semantic feature or establish that a shared/private block-diagonal graph is sufficient. Cross interactions and the remaining function must be retained or priced in an executable approximation. This audit does not overturn earlier cross-patch failures or prove all cross-enabled circuits expensive; its spaces differ from the fitted graph and its complement is not the224-dimensional private branch.

Next decision: use the original-coordinate and covariance objectives as separate constraints when considering any candidate graph. Avoid treating high support overlap as evidence to force more sharing. The completed native graph still fails fidelity; no adoption, OOD, extraction or manipulation claim changes.

Evidence: [architecture control](PAIR_SUPPORT_ARCHITECTURE_V1.json), [interaction energy audit](ALIGNED_INTERACTIONS_V1.json), [audit implementation](audit_aligned_interactions.py). Native factors and registered controls remain linked from the architecture plan.
