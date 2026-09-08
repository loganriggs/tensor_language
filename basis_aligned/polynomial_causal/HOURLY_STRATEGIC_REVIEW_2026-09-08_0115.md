# Hourly strategic circuit review — 2026-09-08 01:15 UTC

## Circuit interpretation targets

1. **Computational specification:** identify what is read, computed, written, and later consumed.
2. **Cross-boundary grouping and within-module splitting:** treat heads and MLPs as candidate boundaries, merging or splitting them according to causal interchange.
3. **Held-out and OOD prediction:** predict unseen inputs, constructions, and task shifts rather than only the discovery examples.
4. **Extraction or sufficiency:** execute the isolated circuit or a precise circuit-plus-background interface and recover its signed effect.
5. **Selective manipulation:** remove, swap, or edit the intended computation while preserving unrelated behavior and measuring redundancy/interactions.
6. **Composition and reuse:** predict joint behavior of shared and task-specific subcomputations.
7. **Stable identification:** survive corpus splits, gauges, and fitting restarts, or use an operational downstream-reader equivalence.

The program goal remains a smaller transparent tensor program that is jointly predictive on fresh/OOD text, composable under joint replacement, selectively manipulable, and literally simpler in storage, compute, edges, state, and program price. Neither low rank nor low reconstruction error substitutes for these circuit criteria.

## What changed since 00:15

The weight-convergence screen strengthened but did not causally establish a common L15H5 reader: construction-specific upstream rank-one coordinates produced distinct residual writes, while all four sources were strongly aligned after the L15H5 Q/K/Q2/K2/V weight maps. The static audit corrected the causal claim because earlier DAS runs replaced complete attention-15 responses and therefore bypassed native L15H5 computation.

The no-refit oracle-by-attention-15 factorial then supplied the missing live-path evidence. All six registered predictions passed in 32 forwards and 2.77 seconds: upstream-only interventions with native attention retained at least 89%/91% of the fixed-attention own-target effect, with direction fraction 1.0 and no worse controls. This establishes a live-attention route candidate, not a particular reader.

The admitted full attention-15 module/nine-head reset-rescue atlas returned a valid null in 120 forwards and 5.04 seconds. Resetting the entire attention-15 write removed only `.057-.081` of the upstream own-target effect, below the frozen `.10` mediation bar. L15H5 was consistently the strongest singleton (`.050-.072`) and singleton effects were nearly perfectly additive, but its v16 effects were unstable (`.031, -.015, .050, -.030`). Therefore the strong weight alignment names a small reader-equivalent side channel, not the dominant causal consumer. Q/K/V splitting and greedy attention-head union are closed. The dominant `.919-.943` residual bypass is the live uncertainty.

The next test was preregistered before implementation: a complete module-write atlas over attention and MLP writes in layers 12--17, with all twelve writes also installed jointly. A reusable absolute-write hook now covers attention tuple outputs and MLP tensor outputs with exact call/shape guards. The thin runner passes focused tests, parse, authority dry run, and experiment gate at exactly 120 differentiable forwards with no fit or backward pass.

## Is this still the highest-information route?

Yes. Complete module writes are the coarsest causal boundary that can distinguish three materially different circuit objects in one cheap run: mediation by one stable downstream module, distributed/additive mediation by several writes, or direct residual carry that survives replacement of every downstream write. It directly advances computational specification, cross-boundary grouping, extraction, selective manipulation, and composition. It does not reopen DAS rank or regularization.

The user-proposed weight translation remains useful after causal localization: once a module or distributed union passes reset/rescue, its write tensor can be pushed through exact downstream weight maps to nominate readers and pulled backward to nominate upstream writers. The L15 result shows why causal admission must precede that translation: static alignment alone can rank a reader that carries only a small fraction of the behavior.

## Confound audit

- **Baseline subtraction:** every vector is a rowwise answer-minus-foil margin relative to the same native base; reset/rescue are compared to their correct upstream-on/off backgrounds.
- **Frame mixing:** A1 and A2 oracle fits remain separate and fixed; parity is held out, and no score pools constructions to hide a failed cell.
- **Nonlinear composition:** margin effects receive exact four-cell Mobius accounting; P/C KL and top-1 flips remain cellwise because KL is nonlinear.
- **Shared token difficulty:** effects are rowwise before aggregation, with eight rows per panel/parity. No examples are dropped.
- **Leakage/post-selection:** all twelve singleton modules and their joint bank are frozen prospectively. One identical singleton must clear every expert/parity cell; no module is selected on A2 or controls after opening.
- **Dead knobs/tripwires:** there are no fit knobs. Authority hashes, terminal dependency, call counts, row coverage, self-clamp replay, exact closure, finiteness, and literal price fail closed.
- **Precision/noise floor:** self-replay uses a fixed-scale logit tolerance of `1e-4`; factorial algebra must close to `1e-12`. The substantive bars (`.10`, `.50`) are far above numerical noise.
- **Ordered replacement:** the joint bank deliberately installs absolute writes captured from one coherent parent execution. Later fixed writes override upstream-induced changes, so this is a causal-scrubbing test of write sufficiency, not a claim that the native modules are independent.

## Genuinely different next routes, ranked

1. **Complete residual-suffix module atlas (current).** Kill it as a mediation route if the joint bank fails `.50` reset and rescue in any expert/parity cell. If it passes and singleton effects compose, run a parity-cross-fit greedy module union; if it passes but does not compose, test interaction-aware module sets.
2. **Residual-state boundary localization.** If the joint write bank fails, patch the carried residual state at successive boundaries (pre/post attention and MLP, layers 12--18) while leaving module writes native. A sharp reset/rescue boundary would identify direct residual transport and the first consuming nonlinear suffix. Kill a boundary claim if effects are diffuse or construction/parity unstable.
3. **Function-space causal-response basis at the first localized consumer.** If native write units remain distributed, identify operationally equivalent write directions by their downstream intervention response, with P/C nuisance responses included as separate axes. Kill it if cross-fit response equivalence does not predict held construction or selective intervention.
4. **Construction-conditioned router.** The stable A1/A2 axes and their large cross-construction angles motivate a finite routed circuit, but both experts currently collide on the same two P rows. Reopen only after a causal downstream site is known and a prospective router discriminator separates target construction from P. Kill it if the router cannot clear controls without using outcomes.

## Throughput and systems audit

From authoritative commits and run logs, the hour produced two new causal receipts: the live-attention factorial (queued 00:25, scored 00:54) and the complete attention-15 mediation null (implemented/admitted during the wait, queued 01:03, scored 01:07). GPU execution itself consumed only 7.81 seconds; shared-lane waiting and scientific implementation dominated serial latency. Claim-to-result latency was about 29 minutes for the dependency factorial and 26 minutes from the 00:41 full-atlas preregistration to result, above the ten-minute basic-screen target. However, these were dependent promotion/localization tests, not independent fast screens, and the reusable capture/scoring/write contracts remove most repeated authoring from the next rung. There was no idle GPU wait after 01:07 attributable to execution; the 8-minute gap to this checkpoint was used for publication, hook construction, and the next runner.

Scientific design and implementation remained the largest work bucket; focused validation and gate work were smaller. The avoidable systems issue is the currently empty queue while the runner is finalized. The immediate repair is to commit and hash-bind the already passing 120-forward runner, enqueue it, and interpret the receipt before opening another bespoke branch.

`CIRCUIT_FOCUS: PASS`

`CEREMONY_BUDGET: PASS`

`NOVELTY_LESSON_GATE: PASS`

The route survives because the next run discriminates downstream write mediation from direct residual carry and because it reuses shared machinery. The next concrete action is managed enqueue of the residual-suffix atlas; its registered result selects either greedy module composition, interaction-aware sets, or residual-state boundary localization.
