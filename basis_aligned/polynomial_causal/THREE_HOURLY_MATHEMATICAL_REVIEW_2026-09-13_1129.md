# Mathematical review — 13 September 2026, 11:29 UTC

## The actual object and the scope mismatch

The target is the reflection-even head9.8 routing component, reached through the head8.2 → MLP8 response. Residual width is 1152, each Q/K map has 128 rows, and the key source basis $B$ has 64 orthonormal columns. Both queries originate in the same residual vector. The two query/key inner products multiply; normalization and BF16-rounded rotary position tables are explicit. The upstream fixed-writer response is rational in edit amplitude. The local unnormalized attention numerator is degree two in query state and degree two in key state, with separate symmetry in the repeated query and key slots. Value multiplication adds another factor; the normalized full program is not a polynomial tensor.

The current rank48 candidate changes only the reflected inside-key numerators. It preserves full Q/K reads, value reads and all denominators. Its amplitude map and adapters are charged: 847,795 stored program scalars versus 871,363 for shared64. These are local compiled dictionaries; dense native context generators and suffix weights remain outside that count. Orthogonal rotation of the retained basis is a gauge: only its projector and executed function are invariant. It does not identify semantic nodes.

Earlier fitting optimized only the inside-inside product, first with independent query ports, then with actual shared query weights and positions folded in. The latter improved the frozen unit-removal screen to 7.17–7.56% response error, but reversed edits reach 15.37%. This is not evidence that normalization has been approximated incorrectly: the two executions use identical normalizers. Normalization can still change the importance of coefficient errors across inputs.

## Exact complete-error algebra

For either QK factor, let $f_i$ be its full normalized score and $i_i$ its inside-key score, using the same full-key denominator. Write $o_i=f_i-i_i$. The exact even routing numerator is

$$
\Gamma=\tfrac12[f_1f_2+(f_1-2i_1)(f_2-2i_2)]
=f_1f_2-f_1i_2-i_1f_2+2i_1i_2
=i_1i_2+o_1o_2.
$$

A reduced projector splits the original inside score as $i_i=p_i+d_i$, retained plus discarded. Because full reads stay fixed, the approximate outside score becomes $o_i+d_i$. Therefore

$$
\widehat\Gamma-\Gamma
=(o_1-p_1)d_2+d_1(o_2-p_2).
$$

**The discarded–discarded product cancels exactly.** Minimizing loss in $i_1i_2$ alone charges a term that is absent from the full error, while omitting cross terms with the outside read. This is a concrete reason to change the fitting object.

[Executed identity control](EVEN_KEY_DISCARDED_PAIR_V1_CONTROL.json): relative error $1.95\times10^{-16}$ on independent scalar-pair probes. A nonzero discarded-only product changes the inside term but causes exactly zero change in the complete even numerator. This is not permission to delete required full-key reads or normalizers.

[Executed response attribution](EVEN_KEY_ERROR_ATTRIBUTION_V1_CONTROL.json): on 72 existing contexts at strengths ±1, the full/full contribution has exactly zero compression error. Mixed and inside-inside error vectors have cosine -0.9954 to -0.9990. Their large cancellation reproduces total scalar error within $1.46\times10^{-15}$. For the failing reversed-edit family, mixed error norm is 0.0561, inside error norm 0.1057 and total error norm 0.0501. The residual is material, but attributing all of it to the isolated inside term is misleading.

## Literature mapping and what it does not solve

- **Tucker/subspace approximation:** [Kolda and Bader's survey](https://www.kolda.net/publication/TensorReview.pdf) distinguishes Tucker subspace representations from sums of rank-one CP terms. Our orthogonal key projector is a tied factor on two key modes of a partially symmetric coefficient tensor. The dense toy verifies the implicit contractions for this restriction. Generic Tucker approximation does not automatically optimize the actual replacement error above: the program preserves full reads while changing the inside/outside split, rather than projecting every occurrence of the key state.
- **Optimization on a Grassmann manifold:** [Absil, Mahony and Sepulchre](https://sites.uclouvain.be/absil/amsbook/) provide the matrix-manifold framework relevant to $P=UU^\top$, $U^\top U=I$. Our QR-retracted gradient solver operates on this representation with a tangent-gradient stopping test. The earlier four-position objective uses small 64×64 Gram matrices, with per-iteration work on the order of the number of positions times $64^3$, plus QR. This gives a practical local solver, not a global optimum or semantic uniqueness guarantee. Ten converged starts agreeing in value remain empirical evidence.
- **Hierarchical Tucker / tensor trains:** [Oseledets and Tyrtyshnikov's hierarchical format](https://epubs.siam.org/doi/10.1137/090748330) concerns compression through mode groupings and low-rank factors. Query-pair/key-pair grouping is a possible representation of our numerator, but it would introduce a different executable graph and must pay for its adapters. No measured small hierarchy rank or normalized-behavior guarantee currently justifies replacing the cheaper bespoke contraction.
- **Hankel realization:** [Rabusseau, Li and Precup](https://proceedings.mlr.press/v89/rabusseau19a/rabusseau19a.pdf) connect weighted automata and linear second-order recurrent networks through Hankel tensors. Our finite, position-dependent normalized attention module is not supplied as such a recurrence with a complete finite-rank Hankel basis. Their realization setting does not currently provide a smaller exact executor here. Constructing a sequence-level realization would be a separate research direction, with additional assumptions and cost.
- **Arithmetic-circuit viewpoint:** [Arvind, Joglekar and Srinivasan](https://arxiv.org/abs/0907.4006) study products of polynomials through circuit representations. The relevant lesson for this object is to preserve multiplication and shared subexpressions when specifying the target. Their setting is not an algorithm guaranteeing minimal circuits for our rational attention program. Here direct distributive algebra supplies the actionable identity without requiring general circuit minimization.

These neighboring formulations are useful distinctions, not interchangeable guarantees. Fully symmetric CP identifiability results do not directly apply to a tensor symmetric only within two mode pairs; equality of representations under a gauge is not circuit identification.

## Decision and next executable consequence

The highest-value next method is a weight-only objective for the **complete replacement error**, retaining the same rank48 program budget. Use the cross-term identity to build implicit coefficient contractions with shared query source and rounded positions. Validate them against a small explicit tensor before fitting. In orthogonal retained/discarded/outside key coordinates, the remaining products connect distinct key blocks; this should avoid materializing a 1152⁴ tensor. The exact contraction cost still needs derivation and measurement before a large fit.

Opposing predictions: if the incomplete target caused the signed-transfer miss, fitting the complete error should reduce signed response errors at matched capacity. If it does not, retain the exact64 program and investigate normalization-weighted or richer block representations; do not erase the original miss or call structure absent. Data should validate the frozen weight fit, not select the subspace.

The present cycle already executes its first consequence: exact discarded-pair cancellation and complete error attribution. Those checks redirect the next objective. The four requested behavioral properties remain incomplete; no new semantic circuit or full-program adoption is claimed. Next mathematical review: 14:29 UTC.
