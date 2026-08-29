# Fixed-projector quadratic closure pilot — preregistration

**Frozen before inspecting real-model leakage values:** 2026-08-29 07:56 UTC

## Question

Do the already frozen, causally evaluated rank-64 early-MLP correction subspaces define
approximately independent quadratic subprograms in the next MLPs?

For an orthogonal projector $P$ and $Q=I-P$, let the symmetric bilinear polarization
of an MLP's quadratic map be

$$
B(x,y)=\frac12D\left[(Lx)\odot(Ry)+(Ly)\odot(Rx)\right].
$$

The direct-sum approximation is

$$
B_P(x,y)=P B(Px,Py)+Q B(Qx,Qy).
$$

Its normalized mixed-block leakage is

$$
\epsilon(P)=
\frac{\mathbb E_{x,y\sim\mathcal N(0,I)}
\|B(x,y)-B_P(x,y)\|_2^2}
{\mathbb E_{x,y\sim\mathcal N(0,I)}\|B(x,y)\|_2^2}.
$$

The Gaussian identity makes this the squared Frobenius error of the corresponding
order-three symmetric coefficient tensors. It is therefore a weights-only tensor
quantity, not an activation-MSE fit.

## Frozen inputs

- Bilin18 checkpoint snapshot `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`.
- Frozen v3 bases artifact
  `basis_aligned/bilinear_quotient/joint_early_mlp_pca_composition_authoritative_v3_bases.pt`,
  SHA256 `0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9`.
- MLP1 is tested with the frozen MLP0-output basis $B_0$.
- MLP2 is tested with the frozen MLP1-output basis $B_1$.
- Both layers are also tested with the QR-orthonormalized union $[B_0,B_1]$.
- Each candidate is compared with two seeded Haar projectors of exactly matched rank.
- Gaussian contraction seeds: `2026082901`, `2026082902`.
- Haar seeds: `1701`, `1702`.
- 64 antithetically paired contraction samples per Gaussian seed, evaluated in
  float32. The two seeds are reported separately; pooling cannot hide instability.

## Predictions and decision

This is an exploratory CPU pilot, not a promotive causal result.

For a candidate to survive:

1. leakage must be at most `0.25` on both Gaussian seeds;
2. leakage must be at most half the median matched-Haar leakage on both seeds, for both
   MLP1 and MLP2;
3. the conclusion must agree for the upstream rank-64 basis and the rank-128 union, or
   the disagreement must be explicitly preserved rather than averaged.

Failure prunes these particular v3 projectors as a tensor direct-sum
canonicalization. It does not prune causal balancing, other projectors, nonlinear
manifold decompositions, or approximate downstream programs. Passing only earns an
exact contraction follow-up and a fresh finite edit; it is not evidence of selective
removal by itself.

