# Equality L5H5 M4 residual-port factor graph V1

This executor moves the 16 frozen residual-to-Q/K projections inside the
no-oracle factor graph.  Its dynamic state inputs are the baseline, child,
remainder, and joint residual corners plus rotary cosine/sine.  It reuses the
four native L5H5 Q/K weight matrices, performs the projections in native BF16,
derives the shared child ports with explicit RMS scales, and executes the four
precision-factor nodes.

There are no raw Q/K activation inputs, oracle scores, fits, or learned
parameters.  The residual corners and their upstream rank-256 M4 producer
remain external, and the arithmetic cost is still 16 native projections.

Integration validation covered 1,811,939,328 projected values on 192 natural
and 192 code-OOD documents: the package and native modules had zero mismatches
and zero maximum absolute error.  Score closure, behavioral replay, all four
node-removal magnitudes, noncopy means, node score norms, and the rolled-node
control also match the raw-port authority exactly.
