# Equality L5H5 M4 raw-port factor graph V1

This executor removes the oracle-score input from the precision factor graph.
It accepts eight raw head ports—derived and native `Q1,K1,Q2,K2`—plus the
rotary cosine/sine context.  It performs head RMS normalization, rotation,
native BF16 factor products, and the float32 explanatory factorization itself.
The native arithmetic residual is computed internally.

The four graph nodes are the first-QK correction, second-QK correction,
algebraic cross-product, and native arithmetic residual.  Their canonical sum
reconstructs the native child score exactly.  The package has zero learned
parameters and no oracle score input.

The declared boundary still includes 16 residual-to-Q/K projections, the raw
derived/native ports, rotary context, and the rank-256 M4 producer.  This is an
extraction improvement rather than a projection-count reduction.

Integration validation on 192 natural and 192 code-OOD documents found zero
score-closure error and exactly zero discrepancy from the validated parent in
every node-removal magnitude, node score norm, noncopy mean, and rolled-node
effect cosine.  Full behavioral replay is likewise exact in every subtype and
half.
