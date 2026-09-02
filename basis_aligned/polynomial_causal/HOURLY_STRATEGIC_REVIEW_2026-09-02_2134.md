# Hourly strategic review — 2026-09-02 21:34 UTC

## Circuit interpretation targets

A useful decomposition must say what information is read, what operation combines it, what is written, and which
later computations use it. It must be allowed to group pieces across native heads or MLPs and split a native module
when its pieces do different jobs. Its claims must predict held-out and shifted inputs, support an executable isolated
circuit or interface, permit selective removal/swap/edit without unrelated damage, compose predictably with other
identified pieces, and remain stable across documents, corpora, fitting restarts, and plausible gauge changes—or be
defined by downstream operational equivalence. Storage and compute matter for the eventual simpler executable model,
but lower rank or fewer values alone is not circuit interpretation.

The full goal remains a smaller transparent tensor program that is jointly predictive on fresh and OOD text,
composable when replacements are installed together, selectively manipulable, and literally simpler in storage,
compute, edges, states, and program description.

## What changed since 20:32

Rung506 found that all 19 later writes are causally live but none has a repeatable 32-circuit fingerprint at the
whole-write level. A receipt-only audit showed that seven MLP writes, including MLP10, do have stable task-context
effects. Rung507 therefore changed the object rather than tuning thresholds: MLP10 is expanded exactly into 253
unordered pairs of its 22 named earlier sources, and named terms will be removed physically before recomputing the
suffix. Gradient attribution is only a no-ranking screen; finite held-out removal and joint-removal prediction are the
identification tests.

The first CUDA smoke opened no scientific effects. It caught a bookkeeping mismatch: the float32 bilinear change is
exact, but comparing it with a difference of separately rounded BF16 writes violates the proposed change-level bound
through cancellation. The repair explicitly accounts for the deployed output-rounding remainder while preserving
the failed pre-repair number. Separately, a CPU audit caught that gradient token counts were being accumulated four
times; this was fixed before any attribution outcome.

## Is this still the highest-information route?

Yes. It directly targets within-MLP splitting, names the earlier inputs on both bilinear branches, tests whether
different terms have the same downstream effect, and measures finite multiple-mediator interactions. A low-rank
approximation could change none of those circuit claims by itself and remains out of scope.

The main confounds are now explicit: score-absent subtraction, BF16 cancellation, nonlinear loss interactions,
shared token difficulty, post-selection from 253 terms, source/gauge instability, and gradient curvature. Fixed
document halves, all-source requirements, retaining every passer, finite confirmation, and joint removals address
them. The current 32-circuit coordinates remain diagnostic only because rung506 falsified them as a stable selector.

## Genuinely different alternatives

1. If 2--8 exact named terms survive, finish finite confirmation and predictive joint composition. This can identify
   split computations and reusable/shared-input structure; it dies if finite effects do not repeat or transfer.
2. If fewer than two survive, learn a coupled dictionary over Left-side source factors, Right-side source factors,
   and output directions, but require its atoms to predict finite interventions. This changes the vocabulary rather
   than reducing rank; it dies if atoms are restart/gauge unstable or causally nonselective.
3. If more than eight survive, add independently defined tasks/circuits that separate their downstream uses. It dies
   if finer task labels do not produce stable groupings on held-out documents.
4. Replace task-loss gradients with a small exact factorial over predeclared source families if gradients fail finite
   confirmation. It dies if family removals are also diffuse or context-unstable.
5. Define a predictive-state quotient from downstream readers: two MLP10 terms are equivalent only when all tested
   later consumers and interventions cannot distinguish them. It dies if the required state dimension remains near
   the raw term count or does not transfer OOD.

The immediate highest-information action remains the repaired no-outcome CUDA smoke, followed by the frozen rung507
run only if every instrument check passes. No rank, quantization, reconstruction, or CE-only result can promote a
circuit claim.
