# Hourly strategic review — 2026-09-09 01:22 UTC

## Circuit interpretation targets

1. **Computational specification:** identify what information is read, the operation performed, what is written, and which downstream computations consume it.
2. **Cross-boundary grouping and within-module splitting:** group causally equivalent pieces across native modules and split native modules when their parts implement different variables.
3. **Held-out and OOD prediction:** predict activation and behavioral effects on unseen text and shifted task constructions.
4. **Extraction or sufficiency:** execute an isolated circuit, or a precisely declared interface plus native background, that reproduces the target signed effect.
5. **Selective manipulation:** edit, swap, or remove the circuit while preserving unrelated behavior and explicitly measuring redundancy and interaction.
6. **Composition and reuse:** predict how a shared computation combines with task-specific branches across tasks.
7. **Stable identification:** retain the same operational units across data splits, gauges, and fitting restarts.

The program-level goal remains a smaller transparent tensor program that is jointly predictive on fresh/OOD text, composable, selectively manipulable, and literally cheaper in storage, compute, edges, state, and program price. Neither low loss nor a completed circuit screen is sufficient.

## What changed since 00:22

- The constrained-DAS diagnosis was corrected: prior Gaussian perturbations were response-space noise on the same task rows, not lexical, constructional, or donor augmentation. A sealed structured-augmentation comparison is now specified.
- An exact linear authority established that the minimum causal intervention rank is $\operatorname{rank}(RV)$ and supplied progressively harder planted toys. Difference in means succeeds only on the rank-one toy. Structured environment augmentation repairs a planted causal/nuisance alias; isotropic noise does not identify the missing direction. A bilinear context gate requires the correctly typed $[x,gx]$ features.
- The notebook-style best-circuit explanation now gives actual prompts and token IDs, shapes, native equations, intervention code, and outputs. It also exposes the scientific gap: the four-head v23 writer is an `is`/`was` temporal circuit with donor caches and native background, not yet a standalone token-to-logit program. Q1/K1/Q2/K2/value routing and the dominant downstream A12--M17 consumer remain unresolved.
- Exact five-factor attention source machinery now decomposes all query/source writes over `q`, `k`, `q2`, `k2`, and effective value, with a causal all-prefix partition into changed temporal tokens, unchanged prefix, and matched suffix. Algebraic closure passes on CPU.
- The first runner audit caught a same-layer overwrite: L9H1 was lost when L9H4 reused the layer-9 cache key. This is being repaired before enqueue and converted into a regression test.

No new GPU circuit receipt landed in this hour because the serial lane remains occupied by v289. The hour produced one exact toy authority, one user-facing computation dossier, one reusable circuit-level factorization primitive, and one prospective factor/source screen. The interval therefore missed the literal one-screen-per-ten-minute receipt target, but the delay was scientific instrumentation plus the already-busy serial lane rather than idle time or repeated bespoke preflight.

## Path decision and confound audit

The input factor/source atlas remains the highest-information CPU-to-GPU route because it directly addresses targets 1, 2, 4, and 5: it asks which input factors and token roles are sufficient for the already-causal four-head write. It dominates another DAS rank sweep or dataset PCA, neither of which would identify the token-to-write computation.

Confounds and controls:

- **Absolute versus additive patches:** the atlas installs absolute pre-`c_proj` head outputs, matching the successful parent intervention. Additive deltas would be evaluated against altered live states at later layers and are not equivalent.
- **Frame mixing:** all 64 frozen v23 rows are retained; target scoring uses the 30 predeclared capable A1/A2 rows and reports both halves separately.
- **Leakage/post-selection:** thresholds and all 32 factor plus 8 source coalitions are frozen. Proper subsets are descriptive only unless they meet the registered selective bars.
- **Nonlinear composition:** Shapley values summarize the measured behavioral coalition game, not raw tensor additivity. Exact tensor closure and parent-replay validity are prerequisites.
- **Source semantics:** source-role arms mix exact attention terms but do not necessarily correspond to one coherent intervened residual state. They identify causal response terms, not a natural-language feature by themselves.
- **Precision/noise floors:** float reconstruction and replay have explicit $10^{-4}$ and $2\times10^{-3}$ gates.
- **Dead knobs/tripwires:** v24 access is forbidden, row and dependency hashes are fixed, exact forward price is asserted, and the caught multi-head overwrite now receives a direct regression test.

Genuinely different alternatives remain: (a) the queued residual/MLP factorial localizes the dominant downstream reader; kill it if no proper suffix interval reproduces the four-head effect; (b) a reader-contracted weight tensor folds the identified write into downstream weights; kill it if it cannot predict held-out interventions; (c) structured augmented DAS tests stable subspace identification; kill it if it cannot beat rank-matched response regression on sealed constructions. The current source atlas survives because none of these alternatives answers which tokens and bilinear attention factors produce the known write.

## Ranked continuation

1. Repair, test, commit, and managed-enqueue the v23 input factor/source atlas. Success requires exact four-head parent replay before any attribution is read.
2. Interpret the already-queued residual/MLP factorial to localize the A12--M17 consumer interval.
3. Join the identified writer factors and downstream readers as restricted weight tensors, then test predictions on held-out interventions before any PCA/SAE description.
4. After v24 native capability, run the sealed structured-augmentation DAS comparison without retuning on v24.

`CIRCUIT_FOCUS: PASS` — every material action advanced circuit computation, identification machinery, or an explicit full-computation explanation.

`CEREMONY_BUDGET: PASS` — validation caught a four-head-to-three-head intervention bug; the focused safeguards are smaller than the 45-forward scientific screen and directly protect validity.

`NOVELTY_LESSON_GATE: PASS` — prior v15 pattern/value and L11H3 QK work were searched; the new screen changes the object to all-prefix, four-head, token-source-resolved five-factor writes rather than repeating either null.
