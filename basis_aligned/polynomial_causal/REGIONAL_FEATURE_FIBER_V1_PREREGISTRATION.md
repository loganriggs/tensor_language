# Four regional current readers plus norm: native MLP16 sufficiency witness

Existing source block supplies two shared quadratic-parent readers and two
child readers. Take their current1152-coordinate parts and form an orthonormal
four-dimensional basis B, without fitting. The two output readers are the
already compiled regional MLP16 contributions including actual block17 scale.
First-source features and downstream17 normalization are outside this local test.

For each quadratic output form S, compute QSB without forming the full1152²
matrix, Q=I-BB^T. Its leading singular vectors give z in span(B), w orthogonal,
and x±=sqrt(1152/2)(z±w). These inputs have identical B^T x and squared norm1152,
yet their normalized bilinear outputs differ by2*1152*sigma/(1+nativeepsilon).
Bias cancels in the difference. No optimization/text fitting or tokenizer input.

A: feature/norm equality relative errors <=1e-12, and direct native folded
quadratic difference matches predicted witness within1e-10 relative error.
B: exact small-feature closure would require cross singular value divided by
norm(SB) <=1e-10 for BOTH output readers. Preserve failure as a certificate
against this candidate all-real-input interface, not approximate token behavior.
This cross-block condition alone is not sufficient for closure: QSQ must also
be isotropic. The helper's positive and anisotropic-complement controls test
that distinction. Another control shows next-state norm need not close even
when an output reader does. No minimum-feature or absent-structure claim.

CPU two matrix-free1152x4 cross blocks and two direct witness pairs; no dense
tensor, native corpus or GPU. Existing MLP16 dossier and prior exact reader-fold
receipt govern the component. This test certifies feature-level intervention
sufficiency, not another spectral rank ladder or causal contribution estimate.
