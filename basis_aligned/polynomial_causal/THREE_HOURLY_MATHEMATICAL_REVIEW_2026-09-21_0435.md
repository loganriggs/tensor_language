# Three-hour mathematical review — 21 September 2026,04:35 UTC

ACTIVE_TRACK: WEIGHT_FOLDING (user-directed two-day focus).

The goal remains circuits with computational specification, stable identification, held-out/OOD prediction, extraction, selective manipulation, composition/reuse and literal simplicity. The last mathematical checkpoint was01:35. This checkpoint includes a fresh primary-literature search and executed consequences, not just a list of analogies.

## Current object and evidence

The original path is last unembedding/MLP reader -> preceding MLP quadratic forms. Native normalized preceding-MLP input z has dimension1152. Two symmetric matrices Qa,Qb give source reads. Their approximations have a common rank-k input dictionary P and two symmetric k-by-k cores Ga,Gb:

    t = P^T z
    qa = ca + la^T z + t^T Ga t
    qb = cb + lb^T z + t^T Gb t
    phi = ((a^T h - qa/2)/s(h) - alpha) (qb/s(h) - beta)
    write = w phi

h is the native last-MLP input and s(h) its original RMS denominator. z's upstream normalization and s(h) stay explicit. Holding these nonpolynomial operations as ports, qa/qb are quadratic and their outer composition contains degrees up to four. This is not a global fixed polynomial of raw tokens. Output w is already in residual coordinates; native final RMS/softcap are used for effects.

The shared16 program costs23588 scalars,16 dense input projections,512 inner transform coefficients,32 source squares and one final product. Shared24 costs33060. All count h-reader and writer but exclude native producers of z,h. Linear basis changes P->PS with inverse core transformations remain gauge freedoms; shared directions are not identified semantic units.

New source/context interchange evidence supports covariance-informed quadratic readers beyond ordinary-state correlations. Shared16 passes the first frozen comparison, but a balanced-donor family fails the relative10% preservation bar at code spaced-word sites (hybrid ratio1.1148, change1.1483). Absolute15%/20% fidelity bars still hold. Therefore no robust adoption claim. Shared24 passes the same numerical comparisons on these opened panels, but selecting it now is post-selection.

## Match 1: HOSVD/common-input projection supplies a useful bound

Map the two covariance-whitened source matrices Mj=Sigma^(1/2) Qj Sigma^(1/2) to a2-by-d-by-d tensor. Leave its output mode uncompressed and tie the two input projections. HOSVD mode truncation provides an error bound but is generally not the best multilinear-rank approximation. [De Lathauwer, De Moor and Vandewalle, Property10](https://www.math.ucdavis.edu/~saito/data/tensor/lathauwer-etal_mulilinear-SVD.pdf).

For this symmetric special case, define K=sum_j Mj Mj^T. Any rank-k common input dictionary has squared coefficient error at least Lk=sum_{i>k}lambda_i(K). Choosing the leading eigenspace V and cores V^T Mj V gives squared error at most2Lk: the row and column residuals each contribute at most Lk. This yields a computable sqrt(2) approximation bound for this restricted coefficient objective, not a global circuit optimum. Dense cost is O(d^3), storage O(d^2); no order-five tensor is formed. Output-rank or activation/semantic uniqueness is not guaranteed.

Executed `audit_shared_source_bounds.py`: against frozen rank16 forms, common-rank8 relative lower bound17.24%, construction17.74%; rank16 lower4.52%, construction6.23%. Against original source quadratic residuals, rank16 lower24.96%, construction25.34%. Thus the original quadratic coefficient residual is not arbitrarily compressible by a16-input dictionary, even though exact mean/linear parts make the full native scalar more accurate. Tiny rank32 residual bounds of order1e-8 are floating-point noise, not a positive exact obstruction.

This bound does not cover arbitrary arithmetic DAGs, empirical fourth-moment loss or native behavioral error. It is a reason not to spend optimizer sweeps trying to defeat a subspace capacity limit.

## Match 2: simultaneous congruence diagonalization tests shared products

Map Ga,Gb to a family of symmetric matrices. If a real invertible X diagonalizes both X^T Gj X, the source forms can reuse the same k squared features rather than evaluating two different sets of k squares. He and Kressner give a generalized-eigenproblem algorithm with exact recovery for SDC families and additional assumptions for robust recovery. Positive-definite guarantees cannot be assumed for our signed cores. [Primary paper, Theorem7 and regularity discussion](https://arxiv.org/html/2402.16557v2).

Executable necessary condition: if Ga is invertible and the pair is real SDC, Ga^-1 Gb has real eigenvalues. `audit_shared_source_pencil.py` checks this with real-diagonal and indefinite-complex planted controls. Shared16 has8 robustly complex eigenvalues, maximum imaginary part0.501, base condition19.9, eigen residual1.74e-15. Thus exact16-common-square diagonalization is obstructed for this pair. Shared8 and24 also have complex pairs. This is a numerical spectral screen with clear margins, not a formal interval certificate.

The relevant restriction is diagonal cores. Mixed products in small blocks remain possible. The literature on simultaneous block diagonalization treats a broader canonical-form problem for symmetric matrix families; it motivates a local block rewrite, not an automatic low-cost guarantee. [Primary block-diagonalization paper](https://arxiv.org/abs/2503.01166). For our pair, the next bounded action is to construct real invariant1-by-1/2-by-2 blocks, verify both congruence residuals and conditioning, and count shared monomials. Cost is O(k^3) for these small cores. Defective/ill-conditioned pencils can invalidate a simple eigenvector construction; a cheap exact rewrite must pass replay before any native claim.

This is preferable to more diagonal-only optimization: a nonzero complex spectrum already rejects that exact target. It preserves existing behavioral failures because an exact graph rewrite cannot repair them.

## Match 3: the paper's M is a moment operator, not merely input covariance

The user's paper defines a functional metric through the expected tensor-product input moments, and gives Gaussian contractions. For degree n, the moment object has order2n; its quadratic case needs fourth moments. [Paper AppendixA.3, equations15–19](https://arxiv.org/html/2605.15183v1).

For a centered Gaussian delta with covariance Sigma, the centered quadratic error has variance2||Sigma^(1/2) DeltaQ Sigma^(1/2)||F^2. This explains our spectral objective after preserving the quadratic mean. Native delta is not Gaussian: its fourth moments and higher-order outer composition are not determined by Sigma. This is why we directly evaluate native scalar and intervention errors. Optimizing exact empirical moments can be done implicitly using probes, but would be data-informed fitting and would require new held-out panels. No claim that covariance alone implements the paper's full functional metric is justified.

## Organization, throughput and next decision

Receipts are in direct_tensor_match; the continuation dossier and packaged shared16 manifest must record the balanced-donor failure, not only the first pass. The canonical overall Logan review remains a high-level trajectory; the0431 report is the current detailed result. Shared executor configuration avoids copying another native runner. Three jobs (48,32,32 captures) each completed in about4seconds; CPU analyses took seconds. Serial authoring and reporting dominate computation. No invented category percentages or independent-agent work are claimed.

Donor concentration was an actual confound: first-choice code donors reused one site44% of the time. The balanced family reduced the maximum to0.285%, revealing a relative-fidelity failure. This changes the adoption ledger and must remain visible.

Best next moves: (1) exact small-block product sharing, which changes program structure without fitting away a failure; (2) fresh control-family confirmation for any subsequently selected wider dictionary; (3) only then deeper upstream closure. The current FineWeb skip11000 cache is exhausted. Attention, native h and normalization dependencies remain explicit. General HT/DAG search and independent-task reuse are still open; neither the current spectral bounds nor canonical-core idea closes them.

Executed consequences: common-subspace lower/upper bounds, common-square pencil obstruction, and balanced-donor sensitivity with paired bootstrap. Goal remains active.
