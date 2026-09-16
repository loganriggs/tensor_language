# Fresh CrossFirst four-stage Hessian correction V1

## Candidate and selection disclosure

Freeze the ordered stage set

`MLP9 + MLP10 + attention17 + final readout`.

Each term is the exact per-context local mixed Hessian applied to the frozen
CrossFirst child/remainder tangents and pulled through the remaining native
suffix.  There are no fitted coefficients: all four gains are one.

This candidate is post-hoc with respect to both earlier Hessian panels.  It was
chosen after observing that energy-ranked `MLP10 + attention17` passed aggregate
composition but missed finite-interaction direction on the first fresh panel.
An exhaustive fixed-width causal minimax audit on the original 96-row allocation
panel ranks this four-stage set first at width four; retrospective performance
on the first fresh panel is motivation only.  Neither earlier panel supplies a
confirmation claim for this candidate.

The sealed authority `CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ROWS.json` contains
48 score-blind prefixes: four new templates, four city pairs, and six new
single-token spelling endpoints.  It had zero full-prefix overlap with every
repository `*ROWS.json` at construction.  This is construction-and-endpoint OOD,
not corpus OOD.

## Intervention and nulls

At the post-attention9/pre-MLP9 boundary use the exact existing child and
remainder removals `a=-child*w`, `b=-remainder*w`.  Measure native, child,
remainder, and joint physical suffix arms.  Propagate the two first tangents and
compute all 18 stage contributions for audit, but sum only the four frozen
candidate stages.

The four one-stage ablations are the preregistered simplicity/necessity nulls:

- omit MLP9;
- omit MLP10;
- omit attention17;
- omit the final readout.

No stage, coefficient, scale, threshold, row, or endpoint may change after the
panel opens.

## Predictions

- **A -- instrument/capability:** all 24 native cue pairs have the expected
  regional contrast sign; baseline replay and complete 18-stage Hessian closure
  are at most `2e-6` and `2e-5` relative error.
- **B -- familywise composition:** in each 12-row family, the four-stage
  corrected joint prediction leaves at most `0.10` of child-effect norm.
- **C -- finite-interaction fidelity:** in each family, the four-stage term has
  relative L2 at most `0.50`, cosine at least `0.95`, and sign agreement at
  least `0.80` against the physical finite interaction.
- **D -- stage necessity:** candidate minimax child-relative error is at least
  10% lower than the best one-stage-ablation minimax error.
- **E -- endpoint transfer:** for each of six new endpoint concepts pooled over
  four templates and both cues, child-relative residual is at most `0.15` and
  finite-interaction cosine is at least `0.80`.
- **F -- control preservation:** every family's unrelated-reader residual is at
  most `0.10` of that family's target child-effect norm; every prompt/stage/
  reader term and physical arm is serialized without filtering.

Only A--F together licenses a fresh four-stage composition rule.  Even a pass
does not identify four standalone semantic modules: native child/remainder
fields, contextual derivatives, and the complete suffix remain charged ports.
Failure is preserved without changing the valid exact-allocation result.

## Price

One checkpoint load; 48 prefixes; one native upstream trace, four physical
suffix arms, and one 18-stage nested-JVP allocation per prefix; zero fitted
coefficients, optimization steps, parameter updates, or backward-to-weight
gradients.
