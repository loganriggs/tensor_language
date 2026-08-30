# Circuit understanding and terminal-evidence rubric

This rubric applies to the canonical circuit registry in this directory.  It does not
alter the schemas or claims of the existing JSON circuit records.

## Mechanistic tiers

- **Tier 0 — candidate.** A behavior and endpoint are proposed, but no controlled
  component intervention localizes them.
- **Tier 1 — causal localization.** Deleting or replacing a named component changes the
  behavior relative to matched component and position controls.  An underpowered result
  or failed null cannot earn this tier.
- **Tier 2 — behavioral characterization.** Tier 1 plus an exact definition of the
  affected tokens, positions, best competitor, effect direction, and magnitude.  The
  effect separates target from matched-negative and off-target positions.
- **Tier 3 — first-order attribution.** Tier 2 plus controlled identification of the
  upstream writers or readers that feed the component.  Marginal ablations remain
  first-order evidence; they are not a compositional explanation.
- **Tier 4 — compositional algebra.** Tier 3 plus an exact replayed algebraic expansion
  of the relevant products of upstream contributions and an executable sufficiency
  test.  A ranking without reconstruction, or an exact expansion whose selected terms
  do not reproduce the behavior, does not qualify.
- **Tier 5 — recursive explanation.** Tier 4 recursively explains the required
  contributions to embeddings, token/position inputs, or other terminal primitives,
  and replays the complete program end to end.

Tiers are cumulative.  `CURRENT tier` means the highest rung already supported by
evidence, never the rung believed attainable.

## Separate terminal-evidence axis

A mechanistic tier is not a terminal certificate.  Extraction, selective removal,
collateral, and OOD transport are recorded separately.  Unless a preregistration
tightens them, the campaign defaults are:

1. Exact source, row, checkpoint, hook census, tensor shape/dtype, and artifact replay
   gates pass before scientific metrics are interpreted.
2. Extraction recovery, relative to a registered deletion/null background, has point
   estimate at least `0.80` and document-bootstrap 95% lower bound at least `0.60`.
3. Selective removal has simultaneous 95% lower bounds above zero for target damage and
   target-minus-matched-negative specificity.
4. Off-target CE upper bound is at most `0.01` nat and at most `10%` of target removal.
   Top-1 change and native-to-intervention KL are always published.
5. Frozen OOD target effect keeps its sign, has lower bound above zero, and retains at
   least `50%` of the in-distribution point estimate.

All arms for a document remain paired through bootstrap.  FIT/SELECT may choose a
program; FINAL and OOD are one-shot and may not choose components, ranks, thresholds,
scales, or behavior definitions.

## Human-readable circuit template

A campaign entry must state:

- current tier and exact claim boundary;
- behavior endpoint, target positions, matched negatives, and collateral denominator;
- exact evidence source/result paths;
- tensor-native form, including discrete selectors and literal execution price;
- extraction and selective-removal interventions;
- OOD split and frozen gates;
- shared-owner/overlap caveats; and
- the smallest experiment capable of promotion.

Component ownership is not exclusive.  Shared parameters and causal CE effects may
support more than one behavior, but are counted only once in a whole-model ledger.
