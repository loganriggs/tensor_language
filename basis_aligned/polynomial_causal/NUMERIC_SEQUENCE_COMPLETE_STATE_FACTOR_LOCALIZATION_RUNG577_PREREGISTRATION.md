# Rung 577 preregistration: numeric-sequence complete-state and attention-factor localization

## Question and claim boundary

The canonical `task.numeric_sequence.continuation` behavior has held FIT and SELECT capability for digit, number-word,
and digit-to-word state shifts.  Changing only the middle value while keeping the final value and answer fixed also
reduces the answer margin, so a final-token successor lookup is not a complete explanation.

This rung asks two causal questions without fitting a subspace or changing any weight:

1. At which fixed final-query boundary does a complete donor state carry digit, number-word, cross-format, and
   relation-dependent behavior while preserving copy and `+2` controls?
2. If the layer-8 H7/H3 output is sufficient, can exact semantic-source score/value terms explain it?

A positive result is site/factor localization only.  It is not a low-rank result, a weight translation, a removal
certificate, FINAL_TEST/OOD evidence, or a complete numeric algorithm.

## Frozen authority

- `increment_two_hypothesis_rows_rung567.json` and receipt: 32 FIT and 16 SELECT semantic groups, with all nine
  numeric-sequence families kept within a group split.
- `numeric_two_hypothesis_capability_rung569_570_results.json` and the independent R571 audit: every required native
  FIT/SELECT endpoint and middle-break gate held.
- `numeric_factor_removal_positions_rung575.json`: authoritative final numeric source/query positions for the five
  overlapping state-shift and copy families.
- R577's model-free semantic-position artifact extends this to the first, middle, and final numeric values for all
  nine families and must reproduce every overlapping R575 mapping exactly.
- Pinned bilin18 checkpoint SHA256
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

FIT is opened first. SELECT is conditional on the rules below. FINAL_TEST and OOD remain closed.

## Families and signed effects

Target-changing interchange is scored separately in both directions for:

- `sequence_digit_state_shift`;
- `sequence_word_state_shift`;
- `sequence_cross_format_shift`.

For each row, the natural source-to-recipient effect is the source-minus-recipient change in the two registered answer
logits.  An intervention's recovery is computed from the **whole cell**: mean intervention effect divided by mean
natural effect, and median divided by median.  There is no row-wise division by a potentially small denominator.

`sequence_middle_value_break` is the relation-dependence family.  In the coherent-to-broken direction, a positive
effect means the patched state lowers the fixed answer's representation-matched numeric margin.  In the reverse
direction, a positive effect means the coherent state restores it.  Recovery again uses whole-cell natural changes.

Active controls are:

- digit and number-word surface-preserving rows;
- digit and number-word copy rows;
- the `+2` conflict, scored by preservation of its native arithmetic-versus-last-successor preference.

Every control must have a nontrivial donor-minus-recipient intervention-vector norm.  A numerically zero edit cannot
pass merely because it preserves the answer.

## Stage 1: complete-state sites

The fixed candidates, in selection order, are:

1. complete L8H7/L8H3 outputs at the final query;
2. complete all-head attention-8 output at the final query;
3. the full 1,152-vector residual after attention 8 at the final query;
4. the full residual after MLP8;
5. the full residual after MLP10;
6. the full residual after MLP12;
7. the full residual after MLP14.

All seven are evaluated on FIT.  The first passing site is selected; a later site cannot replace an earlier passing
site.  The selected site alone is evaluated on SELECT.

A site passes only if every target family/direction has mean and median recovery at least `0.50`, intervention-effect
positive fraction at least `0.75`, bootstrap 95% lower mean effect above zero, and donor answer top among the matching
numeric candidate set in at least `0.50` of rows.  Both middle-break directions require mean/median recovery at least
`0.50`, positive fraction at least `0.65`, and bootstrap lower mean effect above zero.

For every active control and direction:

- median intervention-vector norm is at least `0.10` of the same arm's FIT target median;
- median absolute registered-margin change is at most `0.25` of its FIT target scale;
- median full-vocabulary logit RMS is at most `0.25` of its FIT target scale;
- surface/copy answers remain top among matching numeric candidates in at least `0.75` of rows and mean CE increase
  is at most `0.10` nat;
- the `+2` arithmetic-versus-last-successor preference keeps its sign in at least `0.75` of rows.

## Stage 2: exact below-head semantic alternatives

This stage opens only if complete L8H7/L8H3 passes FIT.  Numeric sources are mapped by semantic ordinal rather than
absolute token position.  The frozen arms are:

- donor attention score with recipient full value at the final source, the first/middle sources, or all sources;
- donor layer-0 cached value with recipient score and layer-8 own value at the first/middle sources;
- donor context-dependent layer-8 own value with recipient score and cached value at the final source, the
  first/middle sources, or all sources;
- joint donor score and full value at the final source, the first/middle sources, or all sources.

They are evaluated on FIT and judged by the same target, relation, and active-control gates relative to complete
L8H7/L8H3. Fixed selection orders exact factors by structural simplicity: final score, non-final score, all scores,
non-final cached value, final own value, non-final own value, all own values, final joint, non-final joint, then all-
source joint. The first passing factor is the only factor eligible for promotion. Every FIT-passing factor is also
evaluated on SELECT for preregistered characterization, but a later factor cannot rescue a failed selected factor or
replace it. No combination is invented after outcomes.

The exact final-source layer-0 cached-value arm is deliberately absent.  R576 already owns its weight-compiled
deletion/sequence characterization.  R576 is a fixed external shared-subroutine comparator; duplicating that arm here
would not distinguish relation state from last-value transport.

## Predictions and decision

- **A:** native replay and semantic-source algebra are exact, with relative squared errors at most `1e-10`.
- **B:** at least one complete-state site passes FIT and the same selected site passes conditional SELECT.
- **C:** L8H7/L8H3 itself passes FIT and SELECT, identifying it as a shared sequence carrier rather than only a list
  carrier.
- **D:** the preregistered structurally simplest FIT-passing semantic factor also passes SELECT.

The strong null is B false: none of the proposed final-query sites jointly carries state shifts, relation dependence,
and selectivity.  C false localizes numeric sequences outside the numbered-list H7/H3 branch.  C true but D false says
the two heads are causal carriers but their numeric-sequence state is not any registered semantic-source score/value
part; compare the complete head with R576 before proposing a new decomposition.

## Price and tripwires

Batch size is 24. FIT has 15 unique-endpoint capture batches and 27 oriented intervention batches; SELECT has 11 and
15. One independent native-replay call is added per opened split. Seven FIT site arms, at most ten conditional FIT
factor arms, one SELECT site arm, and at most ten SELECT factor arms give a maximum price of **652 model forwards, zero
backwards, zero fitted vectors, and zero weight updates**.  Dry-run must recompute this price from the frozen rows.

Any changed input hash, failed R575 overlap, non-single-token semantic value, non-final comma query, dead control,
nonexact replay, nonexact attention source sum/value split, price excess, precondition failure, or access to
FINAL_TEST/OOD invalidates the instrument before scientific interpretation.
