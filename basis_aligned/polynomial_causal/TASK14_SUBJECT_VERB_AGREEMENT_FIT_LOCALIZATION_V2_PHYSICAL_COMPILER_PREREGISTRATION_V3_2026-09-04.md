# Task 14 FIT localization v2: physical compiler v3 repair plan

**Prospective design claim:** 2026-09-04 12:09 UTC. **Status:** CPU-only, outcome-blind design draft. No compiler,
manifest, index, producer, execution, or authorization is frozen by this document.

The v2 compiler commit `6b7fb09ff30080e73cad0414d8315db660e04ca0` remains BLOCKED. This planned immutable v3
successor will retain the approved task14 FIT localization-v2 science while repairing execution-order defects rather
than amending v2.

The compiler must emit calls in causal stage order: native cache, discovery gradients, discovery ceilings, eligible
site joint rank-one fits, diagnostic spectral comparisons, discovery selection, selected family/rank fits, locked
validation, singleton necessity, conditional two-site redundancy, conditional ordered reader, and terminal
projection. Each physical stage will accept only a typed hash-bound receipt from its completed predecessor and only
the decisions available at that point. It cannot accept a future-final active plan when replaying earlier stages.

A pure typed state machine will record every completed, failed, and skipped node. An operational fault after any
completed prefix produces an operational-abort state with no scientific terminal and no package. The DAG transition
table will distinguish such faults from the only publishable `instrument_invalid` cases: fully completed, finite
joint-rank1 or selected-family/rank optimizer/seed-health failures.

Before any stage replay, an exact-global preflight must validate the complete canonical manifest and binary index,
regenerate every descriptor, and return a typed receipt bound to their full hashes and census. Stage replay will
reject a missing or altered receipt, truncated manifest, or synthetic subset. Each active stage replays only its
canonical offset/slices; inactive stages receive explicit zero-call skip receipts.

All runtime values will be checked by exact Python type: booleans cannot be integers or strings; counts and sites must
be integers but not booleans or floats; and site collections must be tuples with exact unique members. Eligible H
count is distinct from the retained top-three H tuple, so more than three causally eligible H sites is valid.

Namespace checks will accept only the three exact frozen, nonempty task14 result/evidence/receipt paths and use
`lstat` so dangling links count as occupied. Deadline checks will accept only the exact 28,800-second ceiling and
reviewed per-stage p99 receipts, track the prior monotonic timestamp, and reject clock rollback or caller-expanded
limits.

The v3 tests will cover causal ordering, future-information rejection, aborts at every prefix, exact-global preflight
binding, direct-replay rejection, strict type attacks, namespace substitutions, deadline enlargement/rollback,
eligible-versus-retained H counts, every previous coherent-policy mutation, and deterministic full regeneration.
The exact compiler artifacts and hashes will be added only after implementation and a fresh parent audit. This plan
does not permit model/checkpoint access, GPU/CUDA, activations, outcomes/results/evidence, queue/enqueue, producer work,
or SELECT/TEST/OOD.
