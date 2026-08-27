# MLP0 quotient Stage-0 v2: deterministic fit-constant amendment

V1 failed closed before any evaluation model forward.  The frozen fit receipt and a
fresh recomputation agreed exactly on response scales, frequency threshold, raw
pre-MLP0 residual-norm threshold, and occupied-cluster counts, but disagreed on token
table and cluster hashes.  A fit-only rediagnostic localized the cause: V1 accumulated
repeated token outputs with CUDA `index_add_`, whose atomic addition order changed the
token table by mean absolute 1.14e-5 and maximum 1.46e-3.  Those small changes flipped
372 Q64 and 573 A64 assignments near k-means boundaries.  No evaluation logits,
losses, response effects, cell statistics, or gates were computed.

V2 makes exactly one outcome-blind construction repair:

- reproduce the historical `mlp0_downstream_clusters.py` CPU token aggregation;
- accumulate the final cluster tables on CPU as a canonical weighted sum.

All other construction, rows, cells, metrics, margins, seeds, gates, and bootstrap
rules are unchanged.  The skip-21000 evaluation role is reused because it remains
outcome-unexposed.  Before freezing V2, two independent fit-only passes were required
to produce byte-identical token tables, Q64/A64 tables and assignments, scales,
thresholds, and occupied counts; this repeatability gate passed.

V1's failure artifact and authority are immutable.  V2 uses distinct fit, authority,
result, failure, and lock paths and refuses overwrite.
