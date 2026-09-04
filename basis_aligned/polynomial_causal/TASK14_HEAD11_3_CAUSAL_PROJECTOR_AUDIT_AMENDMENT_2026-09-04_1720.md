# Task 14 head-11.3 causal projector — final pre-execution audit amendment

Frozen at 2026-09-04 17:20 UTC, before any model load, projector fit, or inner-SELECT score. This closes four fail-open implementation issues found by independent review. It changes no scientific success bar, data split, fitted rank, optimizer, or rank-opening rule.

## Live model-weight integrity

PyTorch parameter version counters are retained as a cheap mutation check, but they are no longer described as the checkpoint hash. The production backend computes a streaming SHA-256 digest over the dtype, shape, order, and current bytes of every frozen live model parameter after load. It recomputes that digest after every fit. A fit is healthy only if both all version counters and the live tensor digest remain unchanged.

## Executable source closure

Program A now pins the exact source bytes of the production backend, the observed-model facade that loads and executes the checkpoint, and the shared projection/interchange library imported by the head adapter. A hash mismatch aborts before fitting.

## Distinct non-identification terminal

A healthy semantic fit whose permutation-label control also passes has not shown a meaningful subject-number variable. That outcome is now `program_a_not_identified`, with an explicit `permutation_control_not_rejected` reason. It is not the registered `small_linear_subspace_null`, which is reserved for the case in which every licensed, healthy semantic fit misses before a provisional rank exists. Fit-health failures remain `instrument_invalid`.

## Exact replay batching

The complete-head SELECT cache and rank-128 replay now use the same SELECT-only order and batch boundaries. FIT controls are cached separately. This preserves the already-registered 1,206-forward / 37,700-example primary ceiling while preventing a bitwise endpoint check from depending on a different CUDA batch layout.

## Data-boundary wording

Program A has a physical **token** boundary: its endpoint shard contains only DISCOVERY token sequences and no VALIDATION tokens or prompt text. It still reads and hashes the full partition and donor metadata authorities to compile the DISCOVERY relation plan. No result may describe this as complete filesystem isolation.
