# Hourly strategic review — 2026-09-03 03:32 UTC

## Circuit interpretation targets

A useful decomposition must eventually provide all seven of these properties:

1. **Computational specification:** say what information is read, what operation combines it, what is written, and
   which later computations use that write.
2. **Grouping across modules and splitting within modules:** combine pieces of different heads or MLPs when later
   computation treats them as one variable, and split a native head or MLP when its pieces do different jobs.
3. **Held-out and OOD prediction:** predict activations and behavioral effects on new documents, task variants, and
   shifted data rather than only reconstructing discovery examples.
4. **Extraction or sufficiency:** an isolated executable circuit, or a precise interface plus background, reproduces
   the target computation or its signed causal effect.
5. **Selective manipulation:** removing, swapping, or editing the circuit changes the intended behavior without
   damaging unrelated behavior, including redundancy and interaction effects.
6. **Composition and reuse:** shared computations serve multiple tasks or modules, and their combined behavior is
   predictable when installed together.
7. **Stable identification:** the units survive corpus splits, plausible gauges, and fitting restarts, or are defined
   operationally by which downstream computations can distinguish them.

The program-level goal remains a smaller transparent tensor program that is jointly predictive, composable,
manipulable, and simpler under literal storage, compute, edge, state, and program prices. Compression, rank, low CE,
or reconstruction alone cannot substitute for these circuit properties.

## What changed since 02:32

- Rung518 produced a valid strong null. All45 attention-head-by-source pieces and990 pairs were causally live, but no
  pair was interchangeable across the32 discovery circuit effects. Two pairs matched four task effects in both
  halves, and the circuit coordinates separated both. This rules out the fixed45-piece vocabulary as a sufficient
  grouping basis without proving that MLP0 lacks simpler computations.
- Rung519 changed the object rather than tuning rank or thresholds. It selects one independently documented circuit,
  `r.2.0.2`, and exactly expands the MLP0 contribution of its stable `H4.DISTANT_SAME` source into47 bilinear
  interactions plus named normalization and deployment-arithmetic terms.
- The full conditional implementation now tests49 finite removals, frozen discovery-to-confirmation transfer across
  all62 distinct circuit masks, and—only for at least two confirmed terms—every finite subset and its exact Möbius
  interactions. The subset experiment directly tests predictable composition and selective removal.
- Three smoke failures/corrections were preserved. The third initial smoke passed. A later full discovery receipt was
  invalid because the float32 fixed-gain identity reached`1.38385e-8` against a`1e-8` exactness bar, although every
  support, identity, liveness, deployed-closure, and final-logit check passed. Its zero-candidate result is not
  evidence. The result and bundle were renamed `invalid_float32_fixed_gain`.
- The narrow repair evaluates the same algebraic terms with float64 accumulation before the unchanged BF16
  interventions. It changes no rows, circuit, term definition, scientific threshold, control, or B--E rule. Thirteen
  focused tests and all static gates pass; a new managed no-outcome smoke is queued.

## Does the current path remain highest-information?

Yes, through the corrected R519 verdict. It directly changes targets 1, 2, 3, 5, 6, and 7: it can identify which exact
input interaction performs one documented circuit's computation, test the identity on new documents/all circuits,
measure all joint interactions, and remove the complete set. The decisive measurements are held-out signed recovery,
target rank and target-to-median specificity across62 circuit masks, subset-profile transfer, whole-atom recovery,
and off-target CE. A valid zero-candidate result kills this source-coordinate refinement; it will not trigger a lower
threshold, larger top-k, rank sweep, or another split of the same source vocabulary.

Rung519 does not yet satisfy extraction or adoption: even A--E would identify a finite interaction program for one
circuit, after which an executable replacement would still have to compose with other identified circuits and earn a
literal price improvement.

## Confound audit

- **Baseline subtraction:** every term and subset is compared with the same native execution. Whole-atom drop is a
  denominator, not a second baseline mixed into the circuit rank.
- **Nonlinear CE composition:** individual CE changes are not added to predict joint changes. All subsets are actually
  run, and Möbius coefficients report the nonlinear interaction.
- **Shared token difficulty:** circuit effects remain member-token CE minus matched-control-token CE. Task CE is
  reported separately.
- **Frame mixing:** all49 terms are writes at the same deployed MLP0 output interface; attention0's direct residual
  write remains native.
- **Leakage and post-selection:** the circuit was chosen from earlier documentation; the atom is selected only on
  R518 discovery; candidate terms use documents500:748; documents752:1000 and the additional30 circuit masks stay
  closed until B. Confirmation cannot add terms.
- **Dead interventions and support:** every term edit must move the MLP0 write; both document halves must contain every
  task category and every circuit member/control cell.
- **Precision:** deployed sum and final logits are checked independently. The invalid float32 receipt is preserved; the
  repair changes arithmetic precision rather than relaxing the frozen`1e-8` bar.
- **Controls:** sixteen per-term permutations of circuit identity retain each term's marginal scale and both halves.
  The real candidate count must strictly exceed their higher-interpolation95th percentile.

## Genuinely different next objects, ranked

1. **Shared attention factor vocabulary across heads.** Jointly decompose Q, K, the second bilinear Q/K branches, and
   value/output factors across attention0 and attention1, but identify factors by their finite downstream use on the62
   circuits. This targets cross-head grouping, within-head splitting, stable identification, and computational
   specification. Kill it if shared factors do not predict held-out attention changes and downstream circuit effects
   better than head-local/permuted controls, or if swapping a factor fails selectively.
2. **Task-defined predictive-state quotient.** Start from the62 known circuit contrasts and group upstream writes only
   when every registered downstream reader treats them interchangeably; then test physical swaps and held-out/OOD
   effects. This directly defines sameness by later computation and avoids assuming head, MLP, or SAE coordinates.
   Kill it if discovery equivalence classes do not transfer or their joint swaps are not compositional.
3. **Later-MLP exact source-pair atlas.** For a documented MLP10 circuit, expand its encoder input against all earlier
   decoder and attention-output sources, use gradients only to screen the finite source pairs, and causally run the
   survivors plus pairwise/higher interactions. This directly addresses multiple-mediator effects. Kill it if the
   gradient screen does not predict finite held-out effects or if no small source-pair set is circuit-specific.
4. **Function-space/Sobolev modes.** Fit shared local input-output functions using values and derivatives, then test
   whether downstream readers identify the same modes across heads/modules. This is a screen unless finite swaps and
   removals validate it. Kill it if modes rotate across splits or do not improve held-out causal prediction over
   activation-only bases.

The descriptive rank-1 MLP0 source-effect result remains useful context: it explains redundancy at an aggregate level,
but it does not group or split circuits and therefore cannot replace any move above. No rank-reduction follow-up is
promoted.

## Immediate action

Finish the repaired managed R519 smoke and full rerun. If A holds and B is false, publish the valid strong null and
immediately preregister move1: a shared attention0/attention1 Q/K/Q2/K2/value factor vocabulary scored by downstream
circuit effects and finite factor swaps. If terms confirm, run the already-implemented subset path before choosing a
new object.
