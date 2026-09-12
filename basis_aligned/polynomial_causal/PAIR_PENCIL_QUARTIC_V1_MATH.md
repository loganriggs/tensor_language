# Joint real blocks for two downstream quadratic forms

12 September 2026, 01:23 UTC. This is a completed weight-only representation screen, not a circuit identification.

The full pair has an accurate shared block representation, but the tested truncation and refit fail native fidelity. This supplies a different exact initialization object; it does not currently improve the compact quartic program.

## Object and relation to earlier work

The current two-output quartic target is

$$
f_m(x)=P(x)^T S_m P(x),\qquad
P(x)=\lambda_{17,0}D_{16}[(L_{16}x)\odot(R_{16}x)].
$$

The two symmetric matrices $S_m$ are the last bilinear layer folded into the fixed centered-unembedding readers. This is one selected two-output group, not the whole unembedding. The actual MLP17 input denominator and surrounding native computation remain external. Bias terms are excluded from this pure producer/producer path.

Earlier full-family congruence tests sought common independent blocks across all token forms and failed their structural bars. The restricted symmetric-LL1 pencil control assumed independent input blocks and rank-one output loading within each block. This test instead allows two arbitrary real symmetric forms to share scalar and two-dimensional blocks. Generic pair structure must not be mistaken for evidence of semantic modularity.

## Restricted numerical construction

Normalize the two forms for the generalized eigenproblem and solve

$$
A v_i=\lambda_i B v_i.
$$

For distinct eigenvalues, symmetry implies

$$
(\lambda_i-\lambda_j)v_j^T Bv_i=0.
$$

Thus real eigenvectors and real spans of complex-conjugate eigenvector pairs give one- and two-dimensional congruence blocks when they form a complete basis. If $V$ is this real basis and $z=V^{-1}P(x)$, then

$$
P(x)^TS_mP(x)=\sum_g z_g^T C_{m,g}z_g,
\qquad C_{m,g}=(V^TS_mV)_{gg}.
$$

Each coordinate of $z$ is itself a quadratic function of the earlier input. Two-dimensional blocks therefore naturally retain products of different quadratic intermediates. The executable readers are rows of the **inverse** basis, not the eigenvectors themselves.

[Lancaster and Rodman (2005)](https://epubs.siam.org/doi/10.1137/S003614450444556X) develop canonical forms for pairs of quadratic forms under congruence over the reals and complexes. Their framework includes more general cases than this implementation. The simple derivation above and our numerical checks do not implement the complete canonical-form algorithm: infinite, defective or repeated-eigenvalue cases may require larger blocks. We check actual reconstruction rather than assuming these conditions from a paper title.

## Executed results and price

The [planted real/complex control](REAL_PAIR_PENCIL_V1_CONTROL.json) recovers two scalar blocks and one two-dimensional block with reconstruction error8.5e-16. The [native pair](NATIVE_PAIR_PENCIL_V1_RESULT.json) has68 scalar blocks and542 two-dimensional blocks. Reconstruction error is4.24e-12, eigen backward error2.72e-16 and real-basis condition842. Numerical and conditioning bars pass.

The full outer representation stores1,333,880 floats, including its dense dual readers and physical writers. It still calls the original MLP16 producer, whose Left/Right/Down matrices alone contain15,925,248 coefficients. These costs cannot be hidden by calling the outer core sparse. Normalization and other boundary inputs remain declared dependencies.

Selecting blocks by individual coefficient energy per reader until32 readers are used gives435.17% pair-matrix error, versus78.94% for independent16+16 spectral truncation. This heuristic misses its comparison bar badly. Large individual block energy can reflect cancellation rather than an important independent computation.

The [refit red-team](NATIVE_PAIR_PENCIL_REFIT_V1_RESULT.json) solves45 allowed scalar-product coefficients per output in the selected blocks. Its error falls to98.94%, but still fails the spectral comparison. Even the unrestricted dense core inside that same32-reader span has97.23% optimal error. Thus fixed amplitudes are not the only problem; the selected span is poor. This does not certify that every possible32-reader block selection is poor.

## Does further backward folding change the verdict?

We executed the [exact producer substitution](NATIVE_PAIR_PENCIL_FOLD_V1_RESULT.json), rather than inferring composed behavior from outer-matrix error. All parameters are fixed from weights before using the developmental cache.

The complete pair representation replays the native two-output quartic write to8.7e-13 relative error. Truncation gives146.13% write error; block refitting gives87.80%. Refit paired-write-change errors are78.19%,81.32%,85.89%,84.53% on agreement verbs, count nouns, past and progressive. Both fidelity predictions fail.

Upstream substitution changes the numerical errors substantially, but does not rescue this selection/refit. No behavioral fit, OOD evidence or semantic reuse claim is made. This screen rules out the tested inexpensive truncation as a current initialization improvement. A different selection criterion or producer-aware factor search remains possible; the ongoing learned-factor fits and mixed-core comparison are separate tests.
