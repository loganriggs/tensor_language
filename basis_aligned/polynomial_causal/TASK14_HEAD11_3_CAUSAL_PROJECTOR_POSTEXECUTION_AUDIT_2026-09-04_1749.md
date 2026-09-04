# Task 14 head-11.3 causal projector: post-execution audit

**Time:** 2026-09-04 17:49 UTC  
**Status:** corrected artifacts pass; scientific instrument invalid

## Goal

The experiment asked whether a small rotated subspace inside attention head 11.3 is enough to transfer the subject-number effect on
Task 14 while leaving answer-preserving controls mostly unchanged. This was a causal interchange test, not a generic low-rank
approximation.

## Computation

For ranks 1, 2, and 4, three independently initialized orthogonal subspaces were optimized on the frozen FIT donor pairs. Each fitted
subspace was then scored on the separate SELECT pairs. The registered run used 1,206 forward passes, 902 backward passes, and 37,700
examples. Rank 8 and all confirmation/permutation work were allowed to open only after a smaller rank passed the earlier gates.

A fit counted as usable only if it was finite, did not update model parameters, reproduced the no-change and full-head endpoints,
remained orthogonal, moved away from initialization, ran all 100 updates, and reduced its objective by at least 0.05 between the first
and final 20 updates.

## Result

All nine fits passed every mechanical check except the required objective improvement. Their improvements were 0.0254–0.0469, below
0.05, leaving zero usable fits at every opened rank. The registered terminal is therefore `instrument_invalid`.

This does **not** show that a small causal subspace is absent. It shows that this fitting method did not converge far enough to make
the subspace test valid. Some target and control scores also missed their bars, but those values are not scientific evidence because
the optimizer-health check failed first. Rank 8, confirmation fits, permutation tests, and Program B stayed closed as required.

## Artifact audit

The first publication mistakenly hashed the float64 reporting view of each frame while storing the same values as float32. That pair
is preserved under filenames containing `artifact_invalid`:

- receipt SHA-256: `e8f9e83ea7df568ce6cef577823d5d6ff96b8b9f95a3b58c94b0b04063d1139d`
- bundle SHA-256: `a858be545bf8e32dd83a9368f4be37a6242bf1c87eedfdf1a860cdffff326134`

The corrected publication hashes the exact stored float32 tensors and passed independent audit:

- receipt SHA-256: `d9576fb38b49976444dd9a4b5e67f43e454838a86e261e251401f1168f22b42b`
- bundle SHA-256: `2fd07b6f052582d81a27b8ce1f9e720b16deadd1c44c46fe4039f8af4eab552d`

All nine frame hashes and the analytic-operator hash match the serialized tensors. All 13 source hashes and the exact execution counts
reconcile, and no validation examples were loaded.

## Consequence

Do not repeat this optimizer unchanged and do not cite the run as a subspace null. The next subspace attempt needs a separately
registered optimizer-health repair or a different mathematically justified identification method. In parallel, the positive causal
path evidence now supports focusing on the internal features used by MLP15 and MLP17.
