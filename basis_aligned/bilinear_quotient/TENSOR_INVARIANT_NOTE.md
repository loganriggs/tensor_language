# A factorization-independent diagnostic for bilinear MLPs

For a bilinear MLP replacement

\[
f(z)=b+\sum_{j=1}^{K} c_j(a_j^\top z)(b_j^\top z),
\]

the observable quadratic coefficient is not `a_j b_j^T`. Reusing the same input on
both legs annihilates its antisymmetric part, so the exact term is

\[
S_j=\tfrac12(a_jb_j^\top+b_ja_j^\top).
\]

Consequently, the factor-only Gram matrix is

\[
G_{jk}=\langle S_j,S_k\rangle_F
=\tfrac12[(a_j^\top a_k)(b_j^\top b_k)
          +(a_j^\top b_k)(b_j^\top a_k)].
\]

If `C` has rows `c_j`, grouping the two symmetric input modes turns the third-order
tensor into a matrix from `Sym²(input)` to output. Its output Gram is exactly

\[
T_{(out)}T_{(out)}^\top=C^\top G C.
\]

The eigenvalues of this 1,152 by 1,152 matrix are squared singular values of the
quadratic tensor unfolding. Rank, stable rank, entropy rank, Frobenius norm, and
mode-energy fractions are therefore invariant under factor scales/signs, input-leg
swaps, component permutations, and any different exact factorization of the same
tensor. This is stronger than counting CP components, while remaining only a
structural diagnostic—not a code length or behavioral score.

This construction is the pairwise-symmetric specialization of the tensor
matricizations and mode spectra in [De Lathauwer, De Moor, and Vandewalle's HOSVD
paper](https://doi.org/10.1137/S0895479896305696). Their paper explicitly treats how
tensor symmetries constrain multilinear decompositions. It does not prove that the
checkpoint CP factors are globally identifiable. The distinction between a tensor
and a particular symmetric rank decomposition is central in [Comon, Golub, Lim, and
Mourrain](https://doi.org/10.1137/060661569), which is why native MLP4 bytes remain a
conditional known-gauge price.

Implementation consequence: compare native and seeded-random candidates at three
separate levels—serialized bits, invariant tensor spectrum, and held-out loss. If a
native program wins behavior only by retaining much larger invariant stable rank,
the result supports tensor capacity rather than semantic alignment. If it wins at
similar bits and similar invariant rank, checkpoint-specific directions carry the
remaining explanatory evidence. Neither conclusion licenses semantic labels without
causal extraction/removal tests.
