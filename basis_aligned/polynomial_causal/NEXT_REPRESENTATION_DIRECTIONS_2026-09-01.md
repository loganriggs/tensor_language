# Next representation-changing directions after the local frontier closure

## Why the next step must change the object

The current goal remains a smaller tensor program that is simultaneously predictive on fresh/OOD text,
composable across replacements, manipulable under named interventions, and literally simpler after every stored
factor, state, index, and dtype is counted.

The overnight work closes the obvious local moves around the adopted program:

- context-Q/K ranks form a measured staircase and the certificate cliff is predicted by a fixed damage ray;
- all 18 MLP input maps are context-low-rank, but exact variable-rank water-filling reselects the adopted
  `{layer0:p768, layer4:p768}` point at its price;
- the next cheaper MLP unit predicts 1.369x its damage, and the direct third cut physically drops 43 to38 certs;
- value ranks64--112 are 3.68--5.48x worse per saved scalar even under an adversarially favorable exponent;
- native CP-atom sharing, identical cross-layer encoders, generic block/tree structure, vocabulary rank/rare-row
  repairs, and token/context finite routers have prospectively failed.

Therefore another Q/K rank, another MLP rank, or another clustering is low-information.  The following directions
change the representation or the information supplied to it.

## 1. Joint context–Sobolev Tucker/CP factorization of a bilinear MLP

For one layer, write the exact quadratic map as

`f(x) = D[(Lx) ⊙ (Rx)] + b`.

The current context-RRR program reduces only the shared input coordinate and leaves the native product width and
output map intact.  A genuinely new program is

`f_hat(x) = U C[(A Qx) ⊙ (B Qx)] + b`,

with input rank `r`, output rank `p`, and product rank `k`:

- `Q`: `r × d`
- `A,B`: `k × r`
- `C`: `p × k`
- `U`: `d × p`

Its literal price is

`P(r,p,k) = dr + 2kr + pk + dp + d`,

versus native `3hd+d` for `d=1152,h=4608`.  Unlike native-atom sharing, this fits new atoms under the live metric
and is invariant to the old factors' permutation/scale gauge.

For scale, `(r,p,k)=(768,768,2048)` would cost `6,489,216` scalars for one MLP, versus native `15,926,400`, a
prospective saving of `9,437,184`.  This is only a price illustration—the predictive and derivative gates decide
whether such a core exists.

Fit a mixed value-and-derivative objective

`sum_x ||f(x)-f_hat(x)||² + λ sum_(x,δ) ||J_f(x)δ-J_f_hat(x)δ||²`,

where the directional derivative is exact for the bilinear layer:

`J_f(x)δ = D[(Lδ)⊙(Rx) + (Lx)⊙(Rδ)]`.

The full embedding population supplies every position-zero MLP0 value exactly; separate natural contextual inputs
and named intervention directions supply the general-regime and Sobolev tests.

First decisive experiment:

1. Recover a gauge-scrambled planted `(r,p,k)` teacher by function, input/output subspace, and derivative transfer.
2. On real MLP0, compare one literally price-matched joint program with context-input-only RRR, output-only PCA,
   and their independently composed product.
3. Freeze ranks from a CPU spectrum/sketch before one physical census build.

Advance only if the joint program lowers two-corpus damage by >=20% versus every equal-or-cheaper control, has
split-stable input/output subspaces, and preserves held-out derivative cosine>=.95.  Kill if it merely matches the
independent input/output composition: then the core contains no extra compressible coupling.

## 2. Vector-valued intervention-aware RRR, not another scalar consequence weight

The earlier signed-response eigenbasis failed because it collapsed downstream consequence to one noisy scalar
gradient.  The certificate study now shows why a richer target is plausible: one universal ray explains about
99.3% of held-out certificate-vector variation, while a stable second residual mode raises R2 to.9991 and makes one
held-out count exact.  That mode was not MLP-specific, so it cannot yet be called a causal module axis.

Construct an augmented metric from a small matrix of named intervention responses instead of a scalar weight.  If
`G_m` maps a layer's local error to the response of intervention/certificate `m`, solve a reduced-rank problem under

`C_aug = C_text + sum_m λ_m G_m^T G_m`.

Equivalently, preserve a suffix-Jacobian subspace alongside ordinary activation covariance.  Use disjoint circuit
groups to fit and test the augmentation; the target is the full signed response vector, not certificate count.

First decisive experiment: at one of layers0 or4 and the same literal p768 price, compare context-RRR, scalar
consequence RRR, and vector-augmented RRR.  Require no worse aggregate CE, >=25% reduction in held-out certificate-
vector residual, split overlap>=.75, and improved original-native signed cosine.  Kill if the augmentation is
split-unstable or merely rotates error between certificate members while leaving the universal intensity unchanged.

## 3. Graph-smooth vocabulary residual instead of rank or sparse exact rows

The input/output vocabulary maps share real predictive geometry, but their remaining error is distributed across
rare tokens: neither 1,129 exact residual rows nor increasing the global residual rank repaired the unseen-token
tail.  A different hypothesis is that this residual is smooth on a graph induced by the already-stored input
embedding, rather than low-rank in Euclidean coordinates or sparse by token id.

Build a deterministic `k`-NN graph from normalized embedding rows, and represent output residual coefficients in a
low-frequency graph-Laplacian or multiresolution diffusion basis.  The graph is not free: price either its edges
and weights, or use a deterministic reconstruction algorithm whose code and working state are included.  Compare
with the same-price independent SVD, shared rank, random graph, and frequency-only bases.

First decisive experiment: fit on the existing 480-row frequency split and evaluate common/rare/unseen bins on two
untouched corpora.  Require >=35% unseen-tail repair and >=20% aggregate improvement over the same-price shared-rank
control on both corpora, with no common-token regression.  Kill if a randomized-degree-matched graph performs
similarly or if graph storage erases the vocabulary saving.

## 4. Asymmetric conditioned contraction algebra for hierarchy or DAG claims

The exact embedding fold identifies the invariant symmetric contraction tensor, but a symmetric quadratic map
cannot orient a DAG.  A legitimate directed object requires ordered information: token substitution direction,
time, distinct input roles, or a named intervention before/after relation.

For a behavior fixed independently of the weights, form conditioned derivative operators

`J_s(x): δ -> T_s(x,δ)`

and test their generated algebra for common invariant flags or approximate simultaneous triangularization.  Blocks
are nontrivial idempotents in the commutant; a hierarchy is a nested reducing flag; a DAG-like partial order needs
the asymmetric operators to be triangular in a stable order.

First decisive experiment: planted block, tree, and directed teachers with gauge scrambling, plus dense and
label-shuffled negatives; then one real behavior whose states each meet support requirements before fitting.
Require planted recovery before interpreting real flags, score subspaces/orders rather than raw units, and include
the finite state/router price.  Kill if the real flag is no more stable than shuffled behavior labels.

## 5. Runtime realization and lower-bit storage after semantic structure stabilizes

BF16 global storage and FP16 Q/K factors are physically free at current resolution, but all measurements dequantize
to fp32.  The next deployment question is separate from semantic compression:

- serialized load time and file size;
- peak host/GPU memory during dequantization;
- fused BF16/FP16 contraction latency and activation memory;
- an int8 or block-quantized physical census/certificate/signed gate.

Do not infer speed from byte count.  A lower-bit arm advances only if its literal metadata/scales are priced and it
preserves the corresponding 62/50/43 tier's predictive and signed gates.

## Ranked order

1. **Joint context–Sobolev Tucker/CP** — largest plausible new semantic saving and closest to structure in the
   bilinear map itself; highest implementation risk, but a strong planted/price-matched falsifier is available.
2. **Vector-valued intervention-aware RRR** — lower implementation cost and directly targets manipulability; kill
   quickly if split stability or held-out response residual does not improve.
3. **Graph-smooth vocabulary residual** — substantial 30M-scale upside, but graph price and unseen-token transfer
   are hard gates.
4. **Asymmetric conditioned algebra** — best route to a defensible hierarchy/DAG, though compression upside is
   conditional on a behavior with independently supported states.
5. **Runtime/lower-bit realization** — high deployment value, but orthogonal to the remaining semantic-scalar goal.

Every new physical candidate should first be projected through the fixed certificate ray.  Under the measured
worst count error of two, a conservative 43-certificate candidate should have a ray-predicted count of at least45;
otherwise it is already beyond the causal budget before GPU construction.
