# Full-input sparse interaction screen

Compare the centered full-unembedding quadratic family of MLP17 with its
MLP16-producer pullback in the paired coefficient metric. No text fitting.
The earlier spectral core kept only128inputs (6.406%total energy before
edge selection). Here retain all1152inputs and all output directions.

For each metric, diagonalize the exact input marginal K and use its complete
orthogonal eigenbasis Q. In that basis the orthonormal symmetric coordinates
are diagonal entries and sqrt(2)times off-diagonal entries. Sum their squared
coefficients over the exact centered output metric. Sorting these energies
gives the exact best k-edge projection **in this fixed frame**, not over frames.
Report k=256,1024,4096 and edges for90%energy; also the top128input restriction.

Producer pullback uses H=lambda^2 D16 G16 D16^T, L'=L17 H^(1/2), R'=R17 H^(1/2).
It is exact for the paired tensor norm, not fully symmetric quartic or text error.
Physical producer readers are H^(-1/2)Q. All outputs are retained via U-centered
Gram Cholesky; no output rank truncation. The common output channel, normalization,
direct residual and other source paths remain separate native background.

- pred_a: CPU dense control, energy/previous-total replay and orthogonality <=1e-8.
- pred_b: composed4096edge capture >=1.25times single-layer4096edge capture.
- pred_c: composed4096edge capture >=0.5.

Failure rejects these fixed spectral-frame concentration predictions only.
If extra inputs do not help, a globally optimized or nonorthogonal frame remains
untested here. This is a reader/interaction discovery screen, not a circuit.
Any promising fixed candidate must later pass native extraction, selective
removal, held-out/OOD prediction and joint edits. Do not tune on prior morphology
examples. Literal conditional cost: d^2+k*d floats,2k edge indices,k product
operations plus reader/writer matrix work; producer native maps/background extra.
Full d=1152,k=4096 cost=6,045,696 floats before upstream/background and common channel.
Streaming uses256edge columns; no vocab-by-d-by-d tensor. Zero body forwards,
zero sequences,900second alarm; managed GPU lane1 only.
