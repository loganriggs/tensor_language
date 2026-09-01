# Prospective removal/composition queue for learned simplicity

Status: active queue design, 2026-09-01 22:49 UTC. This file does not open an evaluation role or authorize an
unregistered model run. Each executable experiment still needs its own frozen preregistration, source gate, and
managed-GPU queue entry.

## Why this queue exists

Rung441 found that the historical candidate manifest has only two signed-removal candidates in one family and no
arm-level composition labels. Rung443 found that a small structural score learned at MLP0 predicts the matched MLP1
candidate ordering, but only inside five already-known grammars. The missing experiment is prospective consequence
prediction on genuinely different program families.

“Dropped” means proposed but not executed. It does not mean abandoned. A proposal leaves this queue only by one of
three routes: it runs; a prerequisite fails and the conditional experiment closes; or a stronger test supersedes it
with the relation recorded explicitly.

## Frozen common consequence schema

For every candidate program `P`, store structure before outcomes are opened:

- complete stored values and bytes, inference operations, interface dimension, and graph-edge count;
- grammar/family, rank or active width, regularization, locality, reuse, conditioning, and native dependencies;
- held-out causal distortion on the program-selection role;
- the frozen rung443 score when its features are in the training support, plus an out-of-support flag.

Removal is a signed intervention. If `Delta_native` is the target behavior change caused by the native intervention
and `Delta_P` is the program's predicted change, record signed-effect error, target-effect cosine, and non-target
collateral. Composition measures a pair chosen before execution: compare the predicted joint change with the actual
joint change, including the interaction term

`interaction(P,Q) = Delta_(P,Q) - Delta_P - Delta_Q`.

Program-fit rows, program-selection rows, and outcome rows remain document-disjoint. Structure and outcome files are
written separately and joined only by hash-bound candidate IDs.

## Ordered experiment queue

1. **Candidate-bank freezer and leakage audit.** Select at least eight priced candidates in each of three physical
   families, freeze candidate IDs and structure, and verify that no removal or composition outcome is present in the
   structural file. This is CPU-only and must complete before any consequence label is generated.
2. **MLP0 folded-program bank.** Use the existing token/state-complete compiler grammar, but score prospectively
   frozen token-only, token-by-context, and context-only signed interventions and preselected candidate pairs. The
   fold is the microscope; the shipped object is still the candidate program, not the folded embedding table.
3. **Attention0 continuous-response bank.** Use the supported rung424/425 continuous response quotient, not the
   rejected convex atoms. Remove registered response coordinates and execute registered pairs through routed U16,
   the six downstream consumers, and suffix cross-entropy. This asks whether continuous coordinates support clean
   edits and predictable interactions.
4. **Late-MLP quadratic bank.** Reuse the MLP16/MLP17 module dossiers to avoid duplicating old quadratic results.
   Select priced quadratic/rank controls and measure the same signed-removal and pair-composition contract on fresh
   rows.
5. **Three-family frozen-score test.** Only after at least 20 candidates across all three families have each outcome,
   test whether the already-frozen structural score orders removal and composition better than bytes, rank, sparse
   edge count, measured selection distortion, and shuffled scores. Do not refit on these labels.
6. **Score-guided new-program search.** Conditional on step5. Search one genuinely new grammar using only training
   data and the frozen score, then compare its sealed consequences with ordinary loss-, rank-, and byte-guided
   searches at matched causal distortion and complete price.

## Explicitly closed or conditional work

- Raw-factor Archetypal Q/K dictionaries are closed by rung439b at the registered512-atom budget.
- The response-metric32-atom/top-4 convex dictionary is closed by rung444. The matched source-permuted convex
  control reconstructs better and the archetypal atoms are less restart-stable than unconstrained atoms.
- Archetypal-atom extraction/removal was conditional on identification and is therefore closed without execution;
  it is not an unfulfilled queued run.
- Optimizing a program under learned simplicity remains queued but conditional on the prospective three-family
  ordering test. Running it earlier would expose the answer key to the objective.

