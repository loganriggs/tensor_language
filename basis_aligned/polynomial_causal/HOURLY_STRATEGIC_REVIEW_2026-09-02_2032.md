# Hourly strategic review — 2026-09-02 20:32 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all of the following:

1. Specify what information is read, what operation combines it, what is written, and which later computations use
   it.
2. Group parts across attention heads or MLPs when later computation treats them as one variable, and split a native
   head or MLP when its parts do different jobs.
3. Predict activations and causal effects on held-out and out-of-distribution data.
4. Extract a sufficient executable circuit or a precise circuit/background interface.
5. Support selective removal, swapping, and editing while preserving unrelated behavior, including redundancy and
   interaction effects.
6. Explain how shared computations are reused and how task-specific branches compose with them.
7. Remain stable across document splits, corpora, fitting restarts, and legitimate gauge changes, or be defined by
   downstream operational equivalence.

The program goal is still a smaller transparent tensor program that is jointly predictive, composable, manipulable,
and simple under literal storage, compute, edge, state, and program price. Rank, quantization, reconstruction, or low
cross-entropy alone cannot satisfy those circuit targets.

## What changed since 19:32

- The sign-gauge discovery was validated: L7H3, L8H3, and L8H4 supply the same copy-score computation when the
  cross-family scale is negated. The reverse L8H4-to-L8H3 action also passes. This groups the score computations up to
  one sign bit, but not their payload/output sides.
- Rung 503's finite singleton screen found that MLP8 carries 25--27% of the local MLP9 response but has the wrong
  loss direction.
- Rung 504 replaced the gradient with exact finite suffix recomputation for all 153 source pairs. The instrument
  passed and the selected set was empty. Actual positive pair copy effects max out around 2--3%, far below the
  registered 10--20% causal floors.
- The complete pair identity `C_st-Q_st=C_s+C_t` recovers all singleton finite loss effects exactly. MLP8 carries only
  `.0064/.0141` of the copy benefit, whereas attention7 is `-.203/-.202`. This closes “MLP9 response equals causal
  mediator” at singleton and pair grain.
- The dossier check prevents repeating work: rungs 465--466 already identified a task-shaped MLP8/9/12 group, a
  broad attention14/MLP17 suppressor group, and their finite interaction on code data.
- Rung 505 is now preregistered and implemented through CPU gates. It tests that fixed program on natural documents
  500--999 under native, positive L5H5, and correctly negated L7H3/L8H3 score supplies, with wrong-sign controls.

## Is rung 505 still the highest-information route?

Yes. The live uncertainty is no longer which raw source makes MLP9 move. It is whether the newly identified upstream
score variable feeds one reusable downstream program or several source-specific realizations. Rung 505 changes the
observation to the already causal five-site program, fixes its sites before natural outcomes, and tests corpus
transfer, source interchange, finite interaction, and sign orientation in one receipt. A pass advances cross-boundary
grouping, composition, and stable identification. A failure distinguishes code-specific grouping from a
source-dependent downstream realization. Both outcomes change the circuit description.

## Confound audit

- **Baseline subtraction:** every action and subset uses the same score-absent trajectory on the same document. The
  reported group effect subtracts two benefits with that common baseline.
- **Frame mixing:** the four score sources are compared only after their action scales are frozen. The L7H3/L8H3
  sources use the validated negative orientation. No normalized-source allocation is used.
- **Nonlinear loss composition:** all 32 five-site subsets are run through the real suffix. Möbius differences are
  computed from finite cross-entropy effects, so interactions are not inferred by summing singleton importances.
- **Shared token difficulty:** source comparisons use the same per-document copy positions. Recovery, per-document
  cosine, and fixed context cells are all reported in two document halves.
- **Leakage and post-selection:** the sites and groups were chosen on the old code role. Natural documents 500--999
  are already open for upstream action calibration, but never for these subset outcomes. No site, subset, or rank is
  selected from rung 505.
- **Dead controls:** native analytical replay, live score edits, live absent-write patches, wrong-sign score actions,
  exact capture/patch counts, and nonzero support are all required.
- **Precision:** the experiment uses deployed BF16 arithmetic and same-document writes. A float32 explanation is not
  substituted for a causal BF16 result.
- **Observation scope:** four interpretable copy contexts are primary. The previously under-supported nine-tag
  circuit battery is deliberately not used as a pass gate; a full 62-circuit transfer remains required later.

## Genuinely different alternatives

1. **Current finite program transfer (rung 505).** Highest information now. Kill it if the fixed natural task group
   loses its signed context pattern, if the code-to-natural transfer fails, or if correctly gauged sources drive
   different task/suppressor/interaction vectors.
2. **Action-conditioned predictive-state quotient.** Cluster downstream states only when all registered future
   interventions give indistinguishable effects. This becomes preferable if rung 505 shows that the fixed native-site
   boundary is code-specific. Kill it if equivalence does not reproduce on held-out actions or if it only reduces
   state dimension without selective interchange.
3. **Within-MLP downstream-effect split across MLP8/9/12.** Contract exact bilinear terms with the finite program
   response across all four score sources, then test exact grouped removal. This is premature until rung 505 says the
   cross-module target itself transfers. Kill it if fixed term groups fail natural and held-out source agreement or do
   no better than matched task-blind controls.
4. **Float32 analytical MLP9 tensor.** It would remove BF16 attribution ambiguity and may explain the local
   cancellation algebra. It cannot repair the observed deployed finite copy null, so it is a secondary mathematical
   explanation rather than the next causal experiment. Kill it as a circuit route if float32 terms do not predict
   finite BF16 removal effects.
5. **Full all-later-write interaction atlas.** This would map interactions beyond the five fixed sites, but the
   combinatorics are large and the current five-site boundary already has a strong prior. It becomes warranted only
   if the five-site transfer fails while the total later correction remains stable. Kill it if interaction estimates
   do not replicate across fixed halves or cannot support selective removal.

## Ranked next moves

1. Run rung 505 through the managed GPU runner and score every clause exactly as registered.
2. If C fails, immediately begin the natural action-conditioned state route; do not refit the five sites.
3. If C passes but D fails, preserve the upstream score gauge and model downstream realization as source-dependent.
4. If A--E pass, use the common finite program response—not rank or activation size—to split MLP8/9/12 internally,
   with exact held-out removal and unrelated-circuit controls.

Rung 505 survives the step-back because it tests an interaction-defined, cross-corpus, gauge-stable computation and
has sharp failure routes. It is not a rank or compression sweep.
