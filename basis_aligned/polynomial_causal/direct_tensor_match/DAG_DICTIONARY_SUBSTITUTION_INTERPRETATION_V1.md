**Replacing an intermediate by a peer combination creates meaningful savings, but the native approximation loses too much fidelity.**

The new graph edit fits a linear combination of peer features to a chosen intermediate, redirects every consumer, and refits the output coefficients. Unlike a single-peer merge, it can recover a low-dimensional feature dictionary whose members are not individually similar. The substitution may remove all the chosen feature's private input projections and products. Global reachable cost is measured after replacement, so reused upstream computations are not falsely charged as removed.

Five planted dictionaries pass: cross-product sum, sum of squares, difference of squares, affine shift and quartic sum. Each has a cheaper program matching fresh artificial probes to1e-10after output refitting. A negative control has a feature within1%of its peers' span but exposes the small independent residual in a downstream contrast. Its output error after substitution/refitting is65.94%, so the edit fails. No equality is inferred merely from small local least-squares error.

On the archived native quartic approximation, each of four quadratic features is a sum of four products. Replacing one by a combination of the other three removes four products and nine thousand input coefficients:

| Representation | Products | Additions | Stored nonunit coefficients |
|---|---:|---:|---:|
|Archived four-feature program|24|44,920|46,096|
|Three-feature substitution + output refit|20|35,711|36,883|
|Same substituted function, exact readout compilation|18|33,393|34,560|

However, the four substituted candidates have calibration errors27.13–40.37%and second-panel errors19.93–30.88%relative to the archived program. All fail the1%fidelity requirement, despite passing the15%product-saving requirement. Only output weights were refit; this does not prove the wider class with learned input directions cannot fit. The archived program itself was an approximation with prior native failures, so even success here would not establish full-model fidelity or a semantic circuit.

The final row illustrates an interaction between topology edits and compilation. Once there are three quadratic features, their pairwise products form a six-element basis. Exact expansion only above that small feature boundary replaces eight root products with six, without expanding the1152-dimensional input tensor. Replay error is4.33e-16, and the selected candidate's27.13%calibration error is unchanged. It saves25%of the original products and about25%of the coefficients, but remains inaccurate.

The same compilation applied to the original four-feature dictionary is worse: it needs ten root products, giving26total products and48,384coefficients. That rewrite should be rejected. Thus compilation must be scored in the actual surrounding graph; one representation is not uniformly cheaper.

These are two new second-stage capabilities: global peer-combination substitution and exact contraction above a small feature boundary. Neither is an arbitrary graph-search solver. The remaining question is whether useful native intermediate dictionaries can be learned or refactored so that these edits preserve the intended computation, rather than merely giving cheaper inaccurate programs.

[Controls](DAG_DICTIONARY_SUBSTITUTION_CONTROLS_V1.json) · [Native substitutions](NATIVE_DICTIONARY_SUBSTITUTION_V1.json) · [Exact compilation](DICTIONARY_READOUT_CONTRACTION_V1.json).
