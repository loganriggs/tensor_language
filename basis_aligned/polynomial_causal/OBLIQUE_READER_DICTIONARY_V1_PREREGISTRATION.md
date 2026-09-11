# Oblique shared-reader dictionaries — registered 11 September 07:36 UTC

Status: numerical cores and CPU controls complete; native runner not yet queued.
Keep the current structured continuation and orthogonal dictionary queue intact.

Question: does a complete nonorthogonal input dictionary organize native
Left/Right weights better than the complete orthogonal alternatives at the same
128coordinates/reader and the same stored-program size? This tests another
structural assumption, with no text or activation fitting.

Use the existing paired3072/1536product splitseed700. Only the6144training
reader directions determine preprocessing and the dictionary. Two preprocessors
are fixed in advance: ordinary uncentered covariance of unit directions, and
Tyler's trace-normalized shape. Their native training estimates are cached in
NATIVE_READER_SHAPE_V1_AUDIT.json; reuse them rather than refitting scatter.
The checkpoint loader reads full weight matrices, but held-out rows do not
enter either estimate. That audit's `heldout_weights_accessed=false` field
refers to fitting use, not literal checkpoint I/O.

Tyler's fixed point is

$$
\Sigma\propto\frac{d}{n}\sum_j\frac{y_jy_j^\top}{y_j^\top\Sigma^{-1}y_j},
\qquad\operatorname{tr}\Sigma=d.
$$

Its shape is invariant to individual nonzero vector rescalings and equivariant
under invertible linear changes up to scale. Existence needs sufficient samples
outside every proper subspace; no ridge or rank truncation is silently added.
See [Kent and Tyler](https://epubs.siam.org/doi/10.1137/0909023) and
[non-asymptotic analysis](https://www.weizmann.ac.il/math/Nadler/sites/math.Nadler/files/publications/23_mestimator_jmva.pdf).
For the proposed population model y=c(s)Ds, with exchangeable coordinate-wise sign-symmetric
nonzero sparse sources s, radial factors cancel and
d E[ss^T/||s||^2]=I. Thus shape proportionalDD^T is a fixed point.
Those population assumptions are not asserted for trained native weights.

Apply the existing MSP solver to unit-normalized whitened training vectors.
For rotation A and symmetric roots S=Sigma^(1/2), W=Sigma^(-1/2), the weight
coder is AW but the input-feature map is AS. They must not be interchanged:
(S A^T)(A W)=I. Normalize synthesis columns to unit norm and rescale codes
reciprocally; the implemented canonical_maps function does this explicitly.

Encoding: choose128indices by largest absolute full coordinates, then solve
the exact least-squares coefficients on that fixed support. This inference
rule applies to training and held-out weight vectors with a frozen dictionary.
It is not a globally optimal support search. Retain native Down/bias and all
background; save sparse codes and the input-feature map, not a dependency on
original dense Left/Right weights for execution.

Four fits: ordinary/Tyler preprocessing x seeds0/937, each max20000MSPsteps and
900soft seconds. Same local stopping bars as the orthogonal fit: relative
tangent gradient<=1e-6 and five-step relative objective change<=1e-9.
Save each fitted map and history, including unconverged results.

Predictions, before native fitting:
- A: full-coordinate and sparse-executor replays<=1e-8; fixed-support normal
  equation residual<=1e-8; all scalar outputs finite.
- B: all four dictionary fits locally converge.
- C: at least one preprocessor's two starts both exceed the best earlier
  orthogonal/identity/PCA held-out normalized-reader capture by .02 absolute.
- D: that same preprocessor's two starts also exceed the best earlier
  orthogonal/identity/PCA full-U coefficient capture by .01 absolute.

Report each arm separately and both two-start family gates. If no family
passes both quality gates, no family is promoted. Baseline results are read
from FULL_READER_DICTIONARY_MSP_V1_RESULT.json after its queued run finishes;
no baseline is refitted or selected to weaken the comparison. All metrics are
weight-based. Prediction C uses1-normalized squared reconstruction error; exact
conditional LS makes it comparable with orthogonal top-k retained energy.

Same program price:7,815,168floating matrix coefficients plus1,179,648indices;
bias, U and other native operations separately charged. Preprocessing roots
are discovery artifacts, not extra runtime adapters once folded into the map.
No native circuit, OOD, selective-removal or timing claim follows from fitting.

Controls: planted oblique recovery meanatomcos.999386; ordinary covariance
also recovers.998895, so Tyler's proposed.005advantage failed. Orthogonal mean
atom alignment for that known dictionary is at most.71349. Native covariance
condition103.75 versus one Gaussianreference6.36; Tyler differs14.42%. Fixed-
support refitting reduces the oblique toy's error62.97%, with exact execution.
These support running the comparison, not a native result. Keep failed gates
and the distinction between covariance geometry and dictionary identification.
