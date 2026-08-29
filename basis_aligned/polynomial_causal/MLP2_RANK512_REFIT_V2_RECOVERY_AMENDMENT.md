# MLP2 rank-512 refit v2 recovery amendment

V1 completed the frozen TRAIN capture and optimization but failed before publishing
its candidate bundle or opening EVALUATION.  The failure was caused solely by a
concurrent repository commit: `HEAD` briefly advanced before `origin/main`, although
all 19 frozen source hashes remained unchanged.  V1 failure and authority are
immutable.

V2 reuses the exact V1 row receipt, source implementation, seed, optimization,
candidate grammar, controls, metrics, gates, and unopened EVALUATION role.  It changes
only the output/lock namespace.  The wrapper must bind the exact V1 failure, prove
`bundle_exists=false`, `evaluation_may_have_opened=false`, and independently bind its
own three recovery source files before delegating to the already audited V1 runner.
No threshold or scientific computation changes.
