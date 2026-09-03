# Rung 582 preregistration: split the cached-value carrier by downstream bilinear use

## Why this follows R576 rather than repeating it

R576 established two different facts.  First, the final visible numeric token contributes an exactly weight-computed
term through layer-8 heads 7 and 3.  Deleting that term damaged all ten numbered-list successor cells and every tested
digit/number-word successor cell.  Second, the deletion was not selective: repeated-list margin and vocabulary-wide
logit changes were about one half of the target scale, and number-word copy exceeded the frozen collateral limit.
R579 independently reproduced the hashes, raw-row summaries, bootstraps, split decision, and null verdict without
loading the model.  The canonical registry therefore says to split the term by downstream read or source/action use,
not to repeat whole-term deletion.

The R582 hypothesis is that R576 found a reusable **numeric-value carrier**, while the distinction between
“advance this value” and “copy this value” appears when later bilinear MLPs combine the propagated carrier with the
rest of the current-query state.  This is a decomposition by computation and downstream behavior.  It is not a rank,
compression, head-importance, or native-unit clustering experiment.

## Fresh source-matched counterfactual groups

The older R567 rows remain authority for the R576 result, but most successor and copy rows used different final
values.  R582 freezes new outcome-blind groups so that action can change while the final source token stays exactly
fixed.  Each group contains two source values, three representations (numbered list, comma-separated digits, and
comma-separated number words), and these cells for each representation and source value:

1. coherent `+1` successor;
2. repeated-value copy;
3. a surface rewrite of each of the preceding two cells;
4. a broken-relation prompt with the same final source and registered `source+1` answer; and
5. a `+2` conflict with both `source+1` and arithmetic `source+2` candidates recorded.

Thus the primary action contrast compares successor against copy at fixed semantic group, representation, source
value, and final source token.  Source generalization compares the two values at fixed action.  Surface cells test
wording invariance.  Relation-break and `+2` rows stop a generic “numeric token present” effect from being called a
successor action.  The prompt, token IDs, exact source/query positions, registered answers, semantic group, and split
are saved before model use.

There are 16 FIT groups and 8 each in SELECT, FINAL_TEST, and OOD.  Words, source ranges, and group identities are
disjoint by split.  FIT opens first; SELECT may open only under the rules below.  FINAL_TEST and OOD stay closed in
this preregistered run.

## Exact downstream computation

Let `c` denote R576's exact cached-value residual term added at the final query after attention 8.  For a candidate
MLP `l` in `{8,10,12,14}`, run two prefix trajectories to its input:

- `r_l+`: native, with `c` present;
- `r_l-`: delete only `c` after attention 8, then let intervening blocks recompute normally.

Apply the model's real RMS normalization separately and define

`x_l+ = rms_norm(r_l+)`, `x_l- = rms_norm(r_l-)`, and `delta_l = x_l+ - x_l-`.

The shipped bilinear MLP is

`M_l(x) = Down_l[(Left_l x) * (Right_l x)] + bias_l`.

Its finite response to the propagated cached-value contrast has the exact expansion

`M_l(x_l+) - M_l(x_l-) = C_l + Q_l`,

where

`C_l = Down_l[(Left_l delta_l)*(Right_l x_l-) + (Left_l x_l-)*(Right_l delta_l)]`

and

`Q_l = Down_l[(Left_l delta_l)*(Right_l delta_l)]`.

`C_l` is the interaction between the propagated cached-value contrast and the rest of that prompt's state;
`Q_l` is the contrast interacting with itself.  The two ordered parts of `C_l` are never interpreted separately,
because swapping the MLP's Left and Right factors is a gauge symmetry.  `C_l`, `Q_l`, and their sum are invariant to
that swap and to the usual per-product rescalings.  Every row/site must reconstruct the direct finite MLP difference
with relative squared error at most `1e-10` before its outcome is usable.

The phrase “action factor” is earned only behaviorally.  A large `C_l` is not automatically an action circuit: it
must distinguish matched successor from copy rows, be stable across source values and wording, and preserve the
active controls below.

## Causal arms and interaction

At each candidate site on FIT, subtract from the native MLP write exactly one of:

- `C_l` (background-cross response);
- `Q_l` (contrast-self response); or
- `C_l + Q_l` (joint finite response).

The suffix recomputes normally.  For a larger-is-better registered margin `Y`, record the exact two-removal
factorial interaction

`I_l = Y(remove C_l and Q_l) - Y(remove C_l) - Y(remove Q_l) + Y(native)`.

This distinguishes a locally additive write decomposition from nonlinear interaction in the remaining model.  It
also prevents the joint arm from being described as the sum of two behavioral effects unless `I_l` is small.

The site/component selection order is fixed: `C_8,Q_8,C_8+Q_8,C_10,Q_10,C_10+Q_10,...,C_14+Q_14`.  The first FIT
candidate satisfying every gate is selected.  Later candidates cannot replace it.  Only the selected candidate and
its two companion components at that site are evaluated on SELECT.

## Measurements and gates

For an answer `a`, the registered margin is its logit minus the largest other candidate of the same representation.
Also save CE, full-vocabulary logit RMS, intervention-vector norm, both registered `+2`-conflict logits, and all
native/intervened logits needed to recompute each statistic.  Every summary is computed from saved per-row records;
group bootstraps resample semantic groups, never individual rows.

A candidate passes FIT, and then independently SELECT, only if all of the following hold:

1. **Exactness and active interventions.** Native replay error and bilinear-response reconstruction error are at most
   `1e-10`.  In every representation/action cell, median intervention norm is at least 10% of the median norm on the
   corresponding coherent-successor cell.
2. **Successor necessity.** In every representation, source level, and surface, at least 75% of coherent-successor
   rows have positive margin damage; the group-bootstrap 95% lower mean margin damage and CE increase are positive.
3. **Matched action separation.** At fixed group/source/representation/surface, define
   `G = successor margin damage - copy absolute margin change`.  The bootstrap 95% lower mean `G` is positive in all
   six representation-by-surface cells and separately at both source levels.  This paired gap is the primary evidence
   that the downstream response distinguishes an action rather than merely carrying a numeric value.
4. **Copy preservation.** In every list/digit/number-word copy cell, the answer remains best in at least 75% of rows,
   mean CE increase is at most 0.10 nat, and median absolute margin change and logit RMS are each at most 25% of the
   corresponding coherent-successor FIT scale.  FIT scales are reused unchanged on SELECT.
5. **Conflict preservation.** The native sign of the arithmetic-`+2` versus last-value-`+1` preference is preserved in
   at least 75% of `+2` rows for each representation/source cell.  This does not assume which candidate the native
   model prefers.
6. **Source and surface stability.** The two source levels agree on the sign of the paired action gap in at least 75%
   of groups, and surface rewrites retain at least 50% of the ordinary-prompt gap with a positive bootstrap lower mean.
7. **Reuse.** All preceding gates must hold separately for lists, digits, and number words.  A list-only pass may be
   reported as a scoped lead but cannot establish the proposed reusable successor-use factor.

Broken-relation rows are a required characterization, not a directional gate.  R576 already showed that the broad
carrier matters when a middle label is broken, and numbered-list behavior can follow final-label-plus-one even when
the preceding relation conflicts.  Therefore low broken-relation recovery would type the selected response as
relation-conditioned, while similar coherent/broken recovery would type it as last-value successor use.  Neither
pattern is allowed to rescue failed action-separation or copy-preservation gates, but imposing one pattern in advance
would incorrectly exclude the other legitimate downstream computation.

R576's saved whole-term effects are reported as a fixed comparator, not used to weaken these thresholds.  In
particular, passing because R576 was even less selective is forbidden.

## Scientific nulls

Two active, norm-preserving nulls are frozen at the FIT-selected site and on SELECT:

1. **Different-group same-cell:** deterministically permute the response vectors across semantic groups while holding
   representation, source level, action, and surface fixed.  The intervention is at the final query, so donor prompt
   length is not a model-coordinate requirement; batching still groups recipient prompts by their own exact length.
2. **Same-source other-action:** apply the matched copy response to the successor row, and the matched successor
   response to the copy row, at fixed group, source, representation, and surface.

Each null must move a nonzero vector with median norm between 0.8 and 1.25 times the real arm.  The real candidate's
minimum action-gap lower bound across representations must exceed each null's corresponding bound.  A dead or
unmatchable null is written as a failed scientific gate; it must not crash or silently disappear.

If no candidate passes, the result is `downstream_use_decomposition_null`: retain R576's broad carrier result but do
not call any MLP response a selective successor circuit.  Do not tune product rank, top-K native units, thresholds,
sites, or source ranges after outcomes.  R577's separate attention score/value localization is complementary and may
run independently; R582 does not duplicate its factor arms.

## Raw evidence and eventual execution price

The eventual result must save, for every prompt/site/arm: row and group IDs; split; all token IDs and semantic
positions; source/action/representation labels; native, source-deleted, and intervened registered-candidate logits;
the full-vocabulary log-sum-exp needed for CE; the squared logit-difference sum and vocabulary count needed for RMS;
`C_l`, `Q_l`, and joint norms; exactness errors; null donor row IDs; and every bootstrap draw or its content-addressed
draw specification.  It must also save the checkpoint and every input hash, opened splits, literal forward count,
zero backward count, and terminal decision.  These are row-level sufficient statistics rather than rounded table
entries, so an independent CPU audit can reconstruct every summary and split-opening decision without storing an
otherwise enormous copy of every full-vocabulary vector.

The dry-run groups prompts by exact token length at batch size 24.  Per FIT batch it prices two prefix captures,
twelve site/component interventions, and two selected-site nulls.  Per SELECT batch it prices two captures, the three
selected-site components, and two nulls.  The generated receipt contains the literal maximum; model weights are never
updated and the planned backward count is zero.  This CPU freeze loads no model and opens no outcome or split.
