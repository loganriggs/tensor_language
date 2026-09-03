# Rung 527 terminal receipt: exact MLP0 context terms do not form reusable circuit groups

**Completed:** 2026-09-03 10:44 UTC  
**Status:** valid strong null  
**Result SHA-256:** `8f8581c2fc0a29ffd45f6383eab9af58d9a239c5715e9467a16843e33f5ee682`  
**Independent audit SHA-256:** `cfc0e5c95dadadaa856c8b080316d89c5e23acbfdb984d5e48fd1a253eb73d2b`

## Question

Does MLP0's context-only computation contain two or more exact source terms that have the same downstream effect on
our known circuits, up to one signed scale? This tests circuit grouping directly. It is not a rank, reconstruction,
quantization, or parameter-count experiment.

The context entering MLP0 was split into five named attention relations: the current token, previous position, nearby
positions, distant positions holding the same token, and other distant positions. For the bilinear map

`B(x,y) = Down[(Left x) * (Right y)]`,

the context-only output was written exactly as five linear terms, five self-interaction terms, and ten unordered
cross-interaction terms. Each quadratic term was centered by subtracting its mean on the frozen fit documents. The
sum of those means was kept explicitly instead of being left in an unnamed remainder.

For each of the 20 terms, the experiment removed that term from MLP0 and ran layers 1--17 normally. Its fingerprint
was the resulting CE change on 32 already defined circuit families. A scale was fit between each pair on the first
document half and had to predict the second half without refitting. Independent permutations of circuit labels were
the null control. The 30 confirmation circuits and both physical term-substitution tests were sealed unless exactly
one to eight pairs passed discovery.

## Instrument result

The instrument passed every exactness and liveness check.

- The 20 centered terms plus the explicitly retained numerical remainder reconstructed the context branch with
  relative squared error `1.66e-23`.
- The numerical remainder held only `0.000753%` and `0.000745%` of context-branch squared energy in the two document
  halves. Rung 517's old unnamed fraction was about `47--52%`; almost all of it was the omitted mean of the quadratic
  terms, not unknown computation.
- All 20 edits changed the deployed BF16 MLP0 output. The smallest write change had RMS `11.49`; there were no dead
  edits.
- Every term had a material downstream CE effect. Term-fingerprint RMS ranged from `0.00880` to `0.05309` nat.
- Exact source partition, state replay, supports, hook calls, and forward counts passed.

## Circuit-grouping result

The registered grouping prediction failed decisively.

- There were 190 possible term pairs. Ninety had enough effect and a finite fitted scale to be eligible.
- Zero of those 90 passed the first document-half relation test.
- Consequently zero passed the second half and zero became candidates.
- All 16 circuit-label permutation controls also produced zero candidates.
- Individual term fingerprints were poorly stable across document halves: median cosine `0.0666`, maximum `0.3012`,
  and minimum `-0.3695`.
- The best descriptive pair was `SELF x NEAR` versus `SELF x DISTANT_SAME`. Its cosine was only `0.550` on the fit
  half and `0.683` on the second half, with relative residuals `0.835` and `0.731`. That is far from the frozen
  equivalence bars.

This is not a low-power or inert-edit null: all 20 terms were large enough to matter, but they did different and
document-dependent things to the measured circuits. No held-out circuit results or physical substitutions were
opened because discovery produced no candidate relation.

## What this establishes

The exact five-relation expansion is useful anatomy and repairs the earlier accounting error. It does **not** provide
a reusable circuit vocabulary at the level of its 20 linear/self/cross terms. Therefore this route is closed after
one valid attempt. We will not tune the thresholds, subdivide the five relations, reduce rank, or claim compression
from the exact reconstruction.

This null does not show that MLP0 is uninterpretable. It shows that a context term's local algebraic identity is not
the same thing as its downstream computational role. A better next object should be defined by how a distributed
state changes later computations: group multiple modules or finer pieces only when their finite boundary-state
changes remain interchangeable under several downstream continuations and on held-out circuits.

## Literal cost

- Full-model forwards: `1,302`
- Backward passes: `0`
- Runtime: `34.25 s`
- Peak GPU memory: `3,159,521,280` bytes (`2.94 GiB`)
- Diagnostic reference values stored: `23,040`
- Deployed values added or removed: `0`
- Compression claim: none

The independent terminal audit recomputed the candidate count, all permutation counts, incremental gates, term
stability statistics, exact hashes, call accounting, and sealed-stage status from the result artifact. All checks
passed.
