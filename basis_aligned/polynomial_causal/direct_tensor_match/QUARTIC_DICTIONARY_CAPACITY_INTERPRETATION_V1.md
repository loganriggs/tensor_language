**Capacity must be checked against the native target, not just the archived approximation.**

For a pure quartic program built from m scalar quadratic features and their pairwise products, the output lies in at most m(m+1)/2directions. This holds regardless of how the quadratic features are learned. A singular-value tail of the target's output-mode unfolding therefore gives a necessary reconstruction-error floor. It does not guarantee that a corresponding quadratic dictionary exists.

For the archived four-feature approximation, a three-feature program has six output directions. Its necessary rank-six floor is3.783%in exact symmetric coefficient Frobenius norm, but only.502%/.315%in the uncentered functional metrics of the two512-row cached panels. The27%substitution error therefore is not forced by output rank on those panels; feature-span restrictions and fitting still matter. Independent dense Gram and native mixing replay controls agree below2e-15.

But the native pure quartic path is broader. Using the saved true native targets on all2048rows per panel:

| Native-target metric | Panel0 | Panel1 |
|---|---:|---:|
|Best possible output-rank6relative error|5.913%|5.715%|
|Best possible output-rank8relative error|5.055%|5.003%|
|Necessary output rank for1%error|414|430|
|Necessary quadratic dictionary width for1%error|29|29|
|Necessary output rank for5%error|9|9|

The dictionary-width bound follows only for the specified pure pair-product readout; additional skip paths, constants, residual outputs or other node types change its capacity and price. These are bounds on the cached native quartic targets, not the full normalized language model and not a universal natural-text error guarantee. The targets are generated directly from MLP16 -> MLP17 -> QR-reduced unembedding weights, excluding residual/bias terms and intermediate/final normalization. Input rows are actual normalized MLP16 states. The source code and hashes remain in the target archive's provenance.

Centering does not remove the broad tail: fitting the mean freely still requires centered output ranks562/588for1%relative variation error, or34quadratic features under the same architecture. These percentages normalize by centered variation energy, so do not compare them numerically as the same error measure as uncentered reconstruction.

The strategic consequence is to retain recent exact/approximate sharing, global substitution and readout-compilation operators, but stop treating smaller versions of an old narrow approximation as progress toward full native fidelity. The five-family graph controls establish machinery, not adequate native capacity. A broader native-target initialization, with width chosen from target-level bounds and all computation priced, is the next relevant decomposition comparison. Rank29is necessary for1%on these panels, not a recommended sufficient width or a promise of identifiable features. Smaller dictionaries may still be useful for explicitly selected circuits; they cannot be advertised as full-target1%replacements.

[Archived-program bounds](QUARTIC_DICTIONARY_OUTPUT_BOUND_V1.json) · [Native-target capacity](NATIVE_QUARTIC_OUTPUT_CAPACITY_V1.json) · [Target provenance](../../bilinear_quotient/ops/run_direct_variation_audit_v1.py).
