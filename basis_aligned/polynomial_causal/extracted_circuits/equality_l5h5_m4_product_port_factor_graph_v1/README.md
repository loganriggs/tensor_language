# Equality L5H5 M4 product-port factor graph V1

This executor composes the rank-256 M4 mode boundary with the residual-port
precision graph.  From the native M4 product tensor and retained `M2,A3,M3,A4`
writes, it constructs the bias-only, rank-128 child, rank-128 remainder, and
rank-256 joint residual corners in canonical BF16 order.  It then performs all
Q/K projections and executes the four exact factor nodes.

The package has no residual-corner inputs, raw Q/K activation inputs, oracle
scores, fits, or learned parameters.  Native M4 Left/Right product generation
and all 4,608 products remain external; this moves the graph boundary upstream
without claiming product-cost compression.

Validation on 192 frozen natural and 192 frozen code documents reconstructed
4,076,863,488 residual-corner values with zero mismatches and zero maximum
error. Score closure, behavioral replay, every node-removal magnitude, noncopy
selectivity, and the rolled-arithmetic control match the residual-port parent
exactly.

An initial invalid run inferred the selected rank from the 1,152-row stored SVD
basis and accidentally admitted modes 256--1,151 into the remainder and joint
corners. Baseline and child were correct, producing a diagnostic near-50%
corner mismatch rate. The executor now requires an explicit selected rank (256
by default), and its unit test uses a larger stored basis than the selected
slice to prevent regression. The failed receipt is retained as
`EQUALITY_L5H5_M4_PRODUCT_PORT_FACTOR_GRAPH_V1_INVALID_STORAGE_RANK_RESULT.json`.
