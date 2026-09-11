# Shared-parent graph move between native LL1 groups

This implements a small part of the proposed arithmetic-DAG search: propose a shared reader between two independently fitted output groups, then jointly refit the reader, private input spaces, and quadratic coefficients against weights. It is not a general graph optimizer or an identified circuit.

Start with two rank16 LL1 quadratics. Select the pair with the largest principal cosine between their input subspaces, which is invariant to internal basis changes. Replace their closest principal directions with a common unit vector u. The other15directions in each frame remain private. Project the two quadratic forms into these new spans, then diagonalize only the private part of each core. The executable result is

$$
z=u^\top x,\qquad y_g=V_g^\top x,
$$

$$
F_{\mathrm{pair}}(x)=\sum_{g=1}^{2}c_g
\left[\alpha_g z^2+2z\,\beta_g^\top y_g+
\sum_{k=1}^{15}\lambda_{gk}y_{gk}^2\right].
$$

One shared parent feeds both output groups. Its square can be computed once. There are31distinct dense readers, two small partner sums,33variable products, and two output directions. The original pair used32dense readers and32squares. Executable coefficient count is38,078versus39,200, saving1,122floats. This counts the reader matrices, alpha/beta/lambda, and output writers; whole-model background and unembedding remain additional. Cached basis/core duplicates are diagnostics, not additional executable fields. Runtime improvement has not been benchmarked.

All discovery/scoring is weight-only under the full unembedding metric. Projection-only selected principal cosines are .994755/.999566. Its squared function changes normalized by whole native tensor energy are3.10e-5/3.48e-6; the first misses the registered1e-5bar, the second passes. The selected parent directions across starts have cosine only.01088, so these maxima do not identify one stable shared variable across runs.

An exact conditional solve of272symmetric-core coefficients barely improves the projected graphs. Both registered1e-6objective-gain predictions miss. This diagnoses the fixed reader choice as the more useful variable to change.

Joint refinement works in the32Dunion of the original input spaces. It learns one normalized parent, two orthonormal private frames, and symmetric cores, keeping output writers and other groups fixed. Native-minus-other-group contractions give the exact reduced objective; the original .01whole-group penalty is retained. No input/data fitting is involved. The union restriction is a computational choice and may exclude better readers outside it.

V1 failed its reduced/full-objective change check: QR flipped private basis signs without transforming the initial cores, so the initial functions differed. Its receipt is preserved and its gain comparison is invalid. V2 uses positive-diagonal QR and verifies initial function preservation before fitting. V2 reduced/full gains replay to1.3e-17, directional gradients to8.8e-11, and graph execution to5.4e-15.

V2 removes16.05%/26.93%of projection-only capture loss. Remaining loss in coefficient-capture fraction is2.6271e-5/2.5755e-6. The first misses the registered25%improvement bar; the second meets its no-regression bar. Firststart reaches500iterations with gradient2.17e-7; second stops on relative reduction with gradient7.57e-9. Neither is a certified global solution, and the first is explicitly iteration-limited. Original projection and gain misses remain unchanged.

The result is a working discrete shared-parent proposal plus continuous weight refinement, with literal graph pricing. It targets reusable computation across LL1 boundaries. The saving is small, the parent is selected after fitting, no broad set of groups has been merged, and there is no fresh/OOD, extraction, selective-removal or behavioral-composition evidence. Next substantive progress should establish reliable multi-group proposals or their stability, not repeat identical optimizer chunks or claim that coefficient overlap identifies a circuit.

Primary receipts: [projection](LL1_SHARED_PARENT_V1_AUDIT.json), [conditional cores](LL1_SHARED_PARENT_REFIT_V1_AUDIT.json), [invalid V1](LL1_SHARED_PARENT_OPTIMIZE_V1_AUDIT.json), [repaired V2](LL1_SHARED_PARENT_OPTIMIZE_V2_AUDIT.json). Sources have matching lowercase names. All runs used CPU only and no corpus access.

The executed follow-up [contribution audit](LL1_SHARED_PARENT_CONTRIBUTION_V1_AUDIT.json) finds that parent-dependent terms account for2.91%of the spectralpair coefficientenergy and48.58%of the native-initializedpair energy. These pairs themselves have0.605%/0.320%of whole-native energy. Shared/private cross terms vanish to numericalprecision because allprivate readers are orthogonal to thecommonparent; this is coefficient geometry, not causal independence on text. Thus the native-initialized merge reuses a substantial computation within its selectedpair, while the spectralparent is relativelyminor. Both executable graphs are now durable namedPTartifacts with hashes/counts in the audit. Their writers are in the full-U isometric1152outputcoordinates; physical residual installation requires the retained inversewhitener and nativebackground. These are executable pair subgraphs, not standalone model replacements.


## Cross-start matching changes the interpretation

Whole-function cosine near 0.9 did not imply stable LL1 blocks. Matching signed group tensors across starts finds only 4 of 64 matches at cosine >=0.8, and one at >=0.9. The tracked native groups 20/62 match spectral groups 18/1 at 0.603/0.244; their shared-parent functions also disagree. These stability predictions fail. Comparing each start's independently selected highest-overlap pair had an additional selection confound. Neither comparison alone excludes a shared computation distributed across different groups. [Group matching](LL1_GROUP_MATCHING_V1_AUDIT.json).

The executed red-team allows that redistribution. The native shared-parent function reaches cosine 0.90156 in the span of the other fit's 64 whole groups, and 0.996389 in the span of its 1,024 individual square components with their fixed writers. Both registered recovery bars pass. [Exact span projections](LL1_DISTRIBUTED_FUNCTION_V1_AUDIT.json).

However, greedy conditional least squares reveals that **one spectral component from group 18 already achieves cosine 0.996387**. The prediction that several groups would be needed fails. Also, 99.6705% of the native shared-parent function's coefficient energy is its squared-reader term; only 0.3295% lies in mixed products. This example therefore exposes a square split between output groups, not a rich hierarchy of multiplicative intermediates. [Support and square/product split](LL1_SHARED_FUNCTION_SUPPORT_V1_AUDIT.json).

Checking the existing dossier and artifacts identifies **atom 150 from the earlier converged 256-square fit**. Its input reader has absolute cosine 0.999108 with the proposed parent, and its full output function has cosine 0.994917. It differs from the previously documented compact shared reader and pronoun quadratic in the tested metrics. The two current square writes have 3.0915 times the energy of their sum, so they partly cancel. Although splitting a term into k equal copies lowers a summed squared-group penalty by 1/k, that generic fact does not explain this particular cancellation. [Alias audit](LL1_SHARED_SQUARE_ALIAS_V1_AUDIT.json).

## Frozen native interface for the previously known square

Freeze the **old** unit reader u of square 150. Its exact native coefficient projection has physical writer

$$
w=D\big[(Lu)\odot(Ru)\big],\qquad
f_{150}(x)=w(u^\top x)^2.
$$

The quadratic matrix uu^T has unit Frobenius norm. Contracting the native tensor with this matrix therefore gives its least-squares output writer. The full unembedding metric is retained. Projection and metric checks pass to 4.5e-14. The old fitted writer has cosine 0.999944 with this exact native writer and 98.69% of its norm. The component contains 0.16999% of total native coefficient energy. This supports a reproducible weight atom across different factorization families, not a new behavioral circuit.

The physical reader/writer pair uses 2,304 coefficients and is stored in STABLE_SQUARE150_NATIVE_INTERFACE_V1.pt; its hash is in the [interface audit](STABLE_SQUARE150_NATIVE_INTERFACE_V1_AUDIT.json). A complete decomposition of MLP17 still requires the native quadratic remainder and Down_bias. RMS normalization, tanh, and the rest of the model remain native.

The strongest negative token writes include quotation marks and apostrophe forms. These are contributions before final normalization; the positive loadings are mixed. No quotation or calibration interpretation has been behaviorally validated. This turn used no text.

Practical consequence: compare proposed DAG nodes across fitting families as executable functions, allow regrouping, and check known atoms before claiming novelty. The clearest shared node here was already present as one square in an older family. General graph discovery and the four behavioral circuit properties remain open.
