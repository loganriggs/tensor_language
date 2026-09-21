**Full quadratic tensor has broad mode spectra; covariance helps inputs more than outputs**

Exact implicit unfolding Grams pass dense controls and native trace/PSD checks. The second managed run completed successfully, reproducing the original folded geometry. Both alternative-geometry low-rank predictions fail.

| Metric | Necessary input rank for10%error | Necessary output rank for10%error | Input512error floor | Output128error floor |
|---|---:|---:|---:|---:|
| Folded concatenation Euclidean |1089|1088|49.66%|80.24%|
| Original residual Euclidean |1118|1090|59.49%|81.21%|
| Historical activation covariance |416|818|8.07%|28.75%|

These are necessary lower bounds for relative symmetric coefficient Frobenius error. Input and output requirements are separate and need not be jointly attainable. Covariance uses2048historical RMS-normalized lastMLP inputs, with centering; no eigenvalues required the declared numerical floor. It is not the empirical fourth-moment function loss, and mean-generated lower-degree terms are outside this homogeneous target.

The result answers part of the original Tucker question: narrow shared input/output ranks cannot reconstruct the full tensor accurately in these metrics, irrespective of the optimizer. At10%error, the folded-Euclidean individually necessary ranks imply a dense symmetric core with645,733,440coefficients before factors. Sparse cores are not excluded by this count; rank can remain high while computation is sparse. Covariance changes the numerical difficulty, but its required output rank remains818.

An actual CPU successor tabulates dense core and factor costs at20/10/5%error for each geometry. This prevents interpreting a lower covariance input rank as evidence of a cheap complete Tucker program. The appropriate next full-target comparison should permit broad output span with structured sparse computation, rather than insist on a small dense global core. Existing native4608channel factorization is an exact such representation; beating its literal graph cost and preserving behavior remains unproved.

No native forward behavior, extraction, feature identity or intervention claim follows from these spectra. The full circuit goal is open.

[Original mode spectra](FULL_TENSOR_MODE_BOUNDS_V1.json) · [Geometry comparison](FULL_TENSOR_GEOMETRY_BOUNDS_V1.json) · [Cost audit](FULL_TENSOR_GEOMETRY_PRICE_V1.json) · [Exact contraction helper](quadratic_mode_grams.py).
