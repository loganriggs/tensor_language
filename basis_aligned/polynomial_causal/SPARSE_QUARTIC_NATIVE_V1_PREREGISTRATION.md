# Sparse quartic native pilot

11 September 2026. User-directed deeper interaction-path decomposition, weights first. Target the full-unembedding homogeneous producer/producer numerator $UQ(P(x,x),P(x,x))$, retaining actual MLP16 and MLP17 weights and block17 residual scale. This is not the complete normalized network or an attention-QK composition.

Use a learned orthonormal $1152\times16$ input bank and 128 selected degree-four monomials out of $\binom{19}{4}=3876$. For a sorted index tuple with multiplicity $M=4!/\prod_i n_i!$, its normalized scalar feature is $\sqrt M\prod_{a=1}^4 b_{i_a}^\top x$. Its exact full-output coefficient is $\sqrt M T(b_{i_1},b_{i_2},b_{i_3},b_{i_4})$. Orthonormality makes these coefficient tensors orthonormal. For fixed bank, retain the 128 largest coefficient-vector energies and solve writers exactly. This is sparse input interaction support with dense output writers, not sparse vocabulary loadings or arbitrary quadratic factors.

Two seeds11511/11512 start from independent random16-dimensional frames in the leading32-dimensional eigenspace of the initialization proxy

$$
L_{16}^\top\operatorname{diag}(\|d_j\|^2\|r_j\|^2)L_{16}
+R_{16}^\top\operatorname{diag}(\|d_j\|^2\|l_j\|^2)R_{16}.
$$

This proxy omits cross-product terms and is not the exact quartic mode Gram. All input directions can move subsequently. These starts explore a common weight-informed region; they do not constitute broad global search.

Optimization alternates exact top-support selection with standard Stiefel Polak–Ribiere CG and backtracking. Each fit has up to20cycles of60softseconds. Scale its objective by its initial selected coefficient energy, a fixed positive constant, so tiny native capture cannot create an artificially tiny stopping gradient. Stop after two unchanged support cycles and relative tangent gradient<=1e-5 with absolute normalized tangent<=1e-7. Maximum time is a pilot budget, not a convergence certificate; preserve unfinished artifacts for continuation. No text, corpus moments, or stochastic optimization probes.

Predictions: **A** native selected-coefficient/oracle agreement<=1e-8, orthonormality<=1e-9, finite outputs and no relative selected-energy decline>1e-8. **B** both arms meet the stated support/gradient convergence criteria. **C** both final energies exceed their respective initial energies by at least10%. These are instrument/optimization screens, not circuit success criteria. Null: finite but unconverged or negligible gain, requiring an optimization or representation diagnosis. Toy controls recovered2/2near starts but0/2cold starts; one cold miss was locally converged. No absent-structure inference is permitted from this pilot.

Literal fitted price:18432reader floats+147456writer floats=165888floats,512integer feature indices,128known multiplicity factors,16linear reads,128quartic outputs. Straight execution uses384scalar multiplies after reading. Compiling repeated quadratic pairs may reduce that count; record the actual graph, do not claim a shared latent computation from syntactic reuse alone. Native upstream/normalization/residual/background dependencies remain charged in a circuit implementation.

The reference norm3.654882852008951e20 is the existing independent Gaussian estimate (estimated relativeSE0.291%). Report selected energy exactly and captured fraction relative to this reference with that limitation. It only scales reporting; optimization is exact for this restricted coefficient objective. A later independent native/OOD panel and selective/composed interventions must follow any frozen candidate. This pilot cannot validate the four properties by itself.

Managed lane1 only; source/dependencies/checkpoint/control hashes bound before enqueue. Dry-run must avoid GPU access. Preserve V1 control syntax failure and V2 repair; the scientific kernel is unchanged.
