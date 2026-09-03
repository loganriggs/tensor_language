# Hourly strategic review — 2026-09-03 04:32 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all seven properties:

1. **Computational specification:** identify what information is read, what operation combines it, what is written,
   and which downstream computations use the write.
2. **Grouping across modules and splitting within modules:** merge pieces of different heads or MLPs when later
   computation treats them as one variable, and split one native component when its parts do different jobs.
3. **Held-out and OOD prediction:** predict activation and causal effects on unseen documents, task variations, and
   shifted data rather than reconstructing discovery examples.
4. **Extraction or sufficiency:** an executable circuit, or a precise interface plus background, reproduces the
   computation or its signed causal effect.
5. **Selective manipulation:** removal, swapping, or editing changes the intended behavior while preserving unrelated
   behaviors, explicitly accounting for redundancy and interactions.
6. **Composition and reuse:** shared computations serve several tasks/modules and their joint behavior is predictable
   when combined with task-specific branches.
7. **Stable identification:** units survive document/corpus splits, plausible gauges, donor choices, and fitting
   restarts, or are defined operationally by downstream equivalence.

The full goal remains a smaller transparent tensor program that is jointly predictive, composable, manipulable, and
simpler under literal storage, compute, edge, state, and program prices. Compression, low rank, reconstruction, and
cross-entropy alone do not satisfy these circuit targets.

## What changed since 03:32

- Rung519's repaired run was valid. Nine of46 exact MLP0 interactions recovered at least15% of one selected source's
  target-circuit effect in both halves and three were stable, but none was top-four or twice the circuit median in both
  halves. Large interactions are shared across circuits rather than specific under this assay.
- The rung519 term-combination probe could fit one half exactly but failed on the other: per-term circuit signatures
  correlated only`.106` across halves and zero of32 fitted combinations localized its intended circuit.
- Rung520 grouped all22 MLP10 terms containing each earlier source into an exact source star. The instrument and83/88
  interventions were live. Two of3,828 pairs matched all four task effects across both halves; zero matched the
  32-circuit pattern even in one complete half. Actual joint removal differed from the sum of singleton removals by
  median relative error8.51 on tasks and9.68 on circuits, directly demonstrating large nonlinear mediator effects.
- A post-result stability probe found median cross-half correlation only`.016` for a source star's own32-circuit
  fingerprint, below a200-permutation q95 of`.077`. The R506--R520 nulls stand as scored, but their broad
  “no structure exists” interpretation is power-bounded.
- A pooled cross-half covariance probe found three positive circuit-effect eigenvalues above a node-permutation null.
  Its top eigenvalue did **not** beat the stronger within-score-action permutation control (`.009331 < .009417`). It
  is therefore a screen for action-level shared response structure, not identified source reuse. It also lives in the
  32-dimensional **circuit-effect space**, not the model's1,152-dimensional activation space; it cannot be installed
  as a DAS projector without a separate localization experiment. “All22 sources are collinear” is stronger than this
  test supports.
- Rung521 is prospectively frozen and pushed. It starts with a whole-attention8 power/stability kill-switch before any
  gradient, because that is the exact intervention object DAS will fit. If powered, it learns a rank4 shared
  projector and orthogonal rank4 private projectors for three historically grouped circuits, with a fourth known
  member held out for reuse. The ranks are fixed capacity, not evidence.
- The mathematical review maps the object to JIVE-style joint/individual orthogonality, Grassmann optimization, causal
  representation learning under interventions, and exact finite Möbius composition. Those methods remove geometric
  gauge and specify valid controls, but none supplies semantic circuit identifiability through the nonlinear suffix.

## Is the current path still highest-information?

Yes, with Stage A fail-closed. Claude's source-star reliability result makes an unconditional DAS fit unsafe; archived
whole-attention8 interchange and mean-ablation effects are far larger and stable in aggregate, so immediately building
a much larger corpus is also premature. Two independent whole-attention8 donor ensembles can decide which case holds
in a few hundred batched executions:

- if whole-attention8 exclusive-member effects and the32-circuit pattern replicate, proceed to the already frozen
  shared/private causal extraction;
- if they do not, stop before optimization and increase donor count or documents rather than interpreting a noisy fit.

This step changes targets2,3,5,6, and7 through exclusive-mask reuse, document/donor transfer, selective private gains,
the full finite factorial, and projector stability. It is not another rank sweep. Even a full pass would still leave
semantic naming and fresh-corpus OOD testing open.

## Confound audit

- **Overlapping circuit masks:** the three fitted masks overlap by21--24%, with77 triple-overlap positions. Primary
  training and gates use cells exclusive relative to all four known cluster masks; the full overlap lattice is only
  reported. Otherwise literal shared tokens could masquerade as a shared computation.
- **Shared token difficulty:** each exclusive cell gets controls from the same parent slice, matched by token,
  position bin, and native-CE decile with a frozen relaxation order. Generic off-slice controls are forbidden.
- **Signed cancellation:** prior DAS trained signed mean damage while reporting absolute damage, and these circuits
  contain mixed help/hurt positions. Rung521 fits and evaluates signed per-token response to the whole-a8 swap, while
  reporting absolute materiality separately.
- **Nonlinear loss composition:** rung520 rules out adding singleton CE effects. Every one of the16 shared/private
  subsets is physically executed and exact Möbius interactions are computed afterward.
- **Donor artifacts:** two four-map donor ensembles, unseen document splits, opposite swap direction, self-donor no-op,
  independent shared/private donors, and mean removal separate a reusable variable from donor-specific steering.
- **Gauge and hidden duplication:** only projectors and principal angles are compared. Shared is learned first; private
  spans are orthogonal and conditional. An unconstrained joint shared/private fit is forbidden.
- **Seed instability and post-selection:** five frozen seeds are required because the historical a8 learned grouping
  had55% relative seed variation. Label permutations retrain the fit; no best TEST seed is chosen.
- **Known-cluster leakage:** `r.2.0.1` was historically known to belong to the cluster. Its reuse result is a held-out
  intervention prediction, not an unbiased grouping discovery. All other a8 circuits are scored so a generic a8
  direction cannot be mislabeled quartet-specific.
- **Stale masks:** two per-circuit JSON files retain old212-row metadata. Only the live1,000-row census masks and frozen
  hashes are authoritative.
- **Multiple mediators:** attention6 is a strong component for all three targets. The final factorial repeats with
  attention6 native and intervened; large dependence changes the name to an `a8 x a6` interaction unit.

## Genuinely different next objects, ranked

1. **Power-gated shared/private attention8 DAS (current).** It directly tests grouping, splitting, reuse, stability,
   and manipulation on a historically supported circuit family. Kill it before fitting if whole-a8 responses do not
   reproduce; kill shared/private claims if exclusive held-out effects, permutation controls, or composition fail.
2. **Rebuild the circuit assay at higher N.** If Stage A fails, sample substantially more FineWeb documents, rerun the
   circuit masks or their classifiers, and measure a reliability curve before grouping. Kill a simple `N` rescue if
   reliability stops improving with document count or the masks do not transfer to fresh text.
3. **Task-defined multi-site predictive state.** Group finite transitions spanning multiple later sites rather than a
   single module's write. This avoids assuming attention8 alone is the semantic boundary. Kill it if state classes do
   not predict new actions/documents or joint substitutions are not closed.
4. **Downstream-reader/tangent localization.** Use the stable pooled circuit-effect modes only as response targets,
   then identify which activation directions and later readers generate them using values and derivatives. This is a
   screen until finite swaps/removals validate it. Kill it if the localized spans rotate across splits or fail physical
   response prediction.
5. **Semantics-first circuit masks.** Improve the weak surface programs for the quartet and test on fresh text before
   claiming OOD semantics. Kill a proposed semantic variable if its classifier fails document/task shifts even when
   the census intervention works.

Optimal singular-value shrinkage or another covariance rank estimate is demoted: it may denoise a response target or
estimate sample size, but by itself it cannot locate, split, extract, compose, or selectively manipulate a circuit.

## Immediate action

Integrate and test the import-safe projector/matching library and the independent Stage-A implementation now in
progress. Freeze its exact control/donor hashes, tighter runtime price, and smoke RMS floor in a preflight addendum;
then run syntax, focused CPU tests, dry-run, static gate, fast suite, and a managed no-outcome CUDA smoke. Only after
that smoke may the managed Stage-A power screen enter the queue. If A passes during the live turn, continue directly
into shared-fit implementation rather than stopping at the gate.
