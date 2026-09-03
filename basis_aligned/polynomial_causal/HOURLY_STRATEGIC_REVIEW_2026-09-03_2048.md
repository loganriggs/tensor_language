# Hourly strategic review — 2026-09-03 20:48 UTC

## Circuit interpretation targets

A useful circuit decomposition must eventually provide all seven kinds of evidence:

1. **Computational specification:** say what information is read, what operation or composition is performed, what is
   written, and which later computations use it.
2. **Cross-module grouping and within-module splitting:** merge parts of different heads or MLPs when later computation
   treats them as the same variable, and split a native module when its parts serve different behaviors.
3. **Held-out and out-of-distribution prediction:** predict activations and causal effects on unseen examples and shifted
   task variants.
4. **Executable extraction or sufficiency:** run the isolated computation, or a precise interface plus background, and
   reproduce its signed causal effect.
5. **Selective manipulation:** remove, swap, or edit the proposed circuit while preserving unrelated computations,
   explicitly accounting for redundancy and interactions.
6. **Composition and reuse:** predict joint behavior when a shared subcomputation is combined with task-specific uses.
7. **Stable identification:** survive data splits, plausible gauges, and fitting restarts, or be defined by operational
   equivalence under downstream readers.

The full goal remains a smaller transparent tensor program that is jointly predictive, composable, manipulable, and
simpler under literal storage, compute, edge, state, and program prices. Lower rank, lower reconstruction error, or
lower cross-entropy damage can price an implementation, but does not itself identify a circuit.

## What changed since 19:48

- R586/R587 repaired the malformed earlier capability artifact and now independently hold the native induction
  selector-by-payload behavior. The audit reproduced 3,024 sequences, 3,240 rows, 108 factorial groups, 432 effects,
  and all 86 registered bootstrap cells. This closes R585's behavioral dependency without rewriting R580/R581.
- R584 ran the exact later-MLP bilinear split. All twelve fixed cross/background, self-interaction, and joint terms were
  active, but none passed every successor-versus-copy selectivity and cross-representation gate. R588 has now
  independently held that FIT-only null from row-level evidence, with 432 bootstrap cells and zero new model calls.
- R589 compared those coarse terms by their causal-response profiles. The numerical correlations reproduce, but an
  independent review caught an overclaim: the thresholds were chosen after FIT outcomes were visible. The only licensed
  result is that no pair passed the recorded post-outcome filter. The MLP12-joint/MLP8-background pair remains an
  unconfirmed prospective hypothesis.
- The R585 replacement design now has an outcome-blind independent approval for implementation. Its model-free package
  fixes exact semantic positions, both A/C roles, all four attention sites, distinct physical units, 20 target cells,
  32 control cells, 24 active-coverage keys, 124 bootstrap cells per split, and a 459+231=690 forward ceiling.
- The first two-agent bootstrap produced reusable tests. A replay can be exactly zero-error yet circular if the same
  faulty term is subtracted and reinserted; later experiments must also reconstruct the canonical term, its remainder,
  and the native output independently. Other promoted checks cover frozen-versus-live timing, same-layer head order,
  ambiguous token positions, omitted components, scale-unit collisions, inactive controls, and split leakage.
- Separate rank work found a roughly 128-dimensional late-MLP write subspace that is inexpensive to share, but §2713
  indicates that it mainly follows common residual-stream geometry rather than a language-model-head readout subspace.
  This is a hypothesis-generating probe basis. It has not grouped behaviors, split MLP tasks, or passed interchange and
  selective-removal tests.

## Is R585 still the highest-information route?

Yes. R584 says a fixed coarse algebraic split of later MLP responses is not selective enough. R585 instead asks a more
specific causal question with opposing predictions: does swapping equality scores move **which source is selected**,
does swapping projected values move **what payload is copied**, and does swapping both reproduce their joint change?
It therefore directly changes targets 1, 2, 4, and 5 if it holds. Its all-four-site form still cannot establish how the
sites divide or duplicate the computation, so a held result would be identification of an operational distributed
factorization, not a complete circuit.

## Confound audit

- **Baseline and nonlinear loss composition:** judge donor-directed logit margins and donor-answer cross-entropy from
  primitive logits; retain the joint interaction rather than summing individual effects.
- **Multiple mediators and live-state drift:** cache native recipient and donor score/value factors at every site before
  intervention. At the intervention pass, subtract the current live equality term and insert the requested frozen
  combination. Earlier edits must not redefine a supposedly untouched later factor.
- **Same-layer order:** L8H3 and L8H4 must be computed from one pre-modification layer-8 state. Sequentially changing the
  state between them would create an unregistered causal path.
- **Circular replay:** verify the isolated equality contribution by both the R459 factor route and the canonical
  attention contraction, and verify `remainder + isolated term = native output` independently.
- **Shared difficulty and broad damage:** require the crossed score-versus-value pattern, donor-answer CE improvement,
  exact no-op identities, small vocabulary-wide change, and preservation on unrelated rows where the intervention is
  demonstrably nonzero. Success or failure shared across hard tokens is insufficient.
- **Counterfactual ambiguity:** keep both physical directions and several constructions. The current dataset varies
  tokens, pair order, filler, and length, and includes selector, payload, joint, match, neutral, filler, and lag edits;
  it remains oracle-equality-supported and in-distribution.
- **Dead controls and unit errors:** intervention-vector norm establishes activity only. Target-margin movement and
  vocabulary-logit RMS use separately frozen scales in their own units, and all rows in an active family enter the
  outcome checks.
- **Leakage and post-selection:** FIT completely decides whether SELECT opens; FINAL/OOD remain closed. R589 cannot be
  treated as confirmation because its filter was written after seeing FIT values.
- **Native-boundary bias:** an all-four-site success does not prove that heads are the semantic units. Site-subset and
  cross-head operational-equivalence tests are required afterward.

## Different routes and kill conditions

1. **Frozen score/value factorial (current R585).** Highest information because the single-factor arms make opposing
   behavioral predictions. Kill the factorization if the complete joint term fails donor-directed transfer, the score
   and payload arms do not separate, or active unrelated controls show a broad contextual write.
2. **Prospective downstream causal-response grouping.** Freeze the MLP12-joint/MLP8-background relationship on a fresh,
   group-disjoint dataset and test whether exchanging one response for the other preserves the same registered consumers.
   Kill it if the profile does not transport or if matched active controls move equally. This stays second because the
   current lead is post-outcome and coarse.
3. **Site-subset operational equivalence after R585.** If the all-four factorization holds, test subsets and replacements
   across L5H5, L7H3, L8H3, and L8H4 to determine duplication, division of labor, and redundancy. Kill any merge when a
   registered continuation distinguishes the sites.
4. **Predictive-state causal quotient.** Group internal states only when a separating family of downstream continuations
   and interventions gives the same output distribution. A single counterexample continuation kills a proposed merge.
   It is more general but currently lacks a tractable complete separating set.
5. **Task-conditioned shared late-MLP core.** Use the discovered shared subspace only as a candidate coordinate system;
   fit or select directions by distinct circuit counterfactuals and test interchange/removal. Kill it as a semantic unit
   if task effects do not cluster or active controls show generic residual-stream dependence.
6. **Rank, PCA, and Fisher price maps.** Retain only as matched-capacity controls or post-identification implementation
   prices. They are demoted because the measured quantities alone change none of the seven interpretation targets.

## Ranked next actions

1. Finish the R585 runner, owner tests, and deterministic dry run; reconcile them against the independent critic's exact
   obligations before any model call.
2. Freeze those exact bytes in a clean commit and give a reviewer the committed runner—not its outcomes—for a second
   implementation-level audit. Enqueue through the managed runner only if every causal and bookkeeping invariant holds.
3. Interpret FIT before SELECT opens. A failed complete-joint arm ends this factorization at these four sites; a held
   factor split advances to site-subset operational equivalence and weight translation.
4. Convert both wave-1 knowledge packets into reusable helpers, planted failures, and a versioned prompt. Then assign two
   different circuit hypotheses in wave 2 rather than duplicating the same tool ambiguity.
5. Keep the prospective MLP12/MLP8 response hypothesis as a second circuit lane, but do not spend fresh data until its
   intervention and opposing prediction are written without reference to R584 FIT outcomes.

The causal direction still dominates on expected information. The selected action is already active: one agent is
implementing the exact R585 runner, the independent specification review and tests are committed, and parent is
reviewing the resulting bytes before managed execution.
