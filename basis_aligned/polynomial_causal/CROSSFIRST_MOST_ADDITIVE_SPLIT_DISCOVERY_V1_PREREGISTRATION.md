# CrossFirst consumer-defined most-additive split discovery V1

## Question

The current CrossFirst child and remainder are producer-defined scalar fields
along the same residual direction.  They are not balanced: on the preceding
fresh panel the child has only about 4% of the total behavioral-effect norm.
Test the DCT briefing's most-additive-split proposal without using the suffix or
behavioral outcomes for selection.

For a frozen scalar `lambda`, repartition the unchanged parent edit as

`child_lambda = child + lambda * remainder`

`remainder_lambda = (1 - lambda) * remainder`.

The two fields still sum exactly to the same parent and require no new learned
vector.  Scan the fixed grid `lambda = 0,.025,...,.75`.  At the first downstream
consumer only, evaluate

`g(z) = z + MLP9(RMS(z))`

and choose the feasible lambda minimizing the worst-family finite interaction
norm divided by the smaller single-piece effect norm.  Feasibility requires,
in every selected family, child effect at least `.50` of total local effect,
remainder effect at least `.25` of total, and neither above `1.25` of total.
This rules out the trivial `child=parent, remainder=0` solution and implements
the briefing's ≥50% retained-effect constraint locally.

Use the 48 already-open `CROSSFIRST_HESSIAN_TOP2_FRESH_V1` rows for discovery.
Repeat the same scan for four deterministic equal-norm writer directions
orthogonal to the real writer.  No downstream suffix, logits, answers, Hessian
allocation, or behavioral outcome may enter selection.

## Predictions

- **A — instrument:** native MLP9-stage replay and every input partition closure
  are at most `2e-6` relative error.
- **B — nondegenerate candidate:** at least one grid point is feasible in all
  four families.
- **C — local additivity:** the selected split lowers worst-family interaction
  over smaller-single-effect by at least 30% from the original split and reaches
  at most `.10`.
- **D — stability:** every leave-one-family-out selection is within `.05` of the
  full-panel lambda, and the full-panel candidate is feasible on the omitted
  family.
- **E — direction specificity:** the real writer's selected minimax interaction
  is at least `.02` below the median selected minimax of the four orthogonal
  equal-norm writer controls.
- **F — audit:** serialize rowwise norms for every direction/lambda, all family
  reports, selections, feasibility masks, and exact price.

If A fails, repair mechanics before interpretation.  Failure of B means this
one-parameter hierarchy-preserving family cannot produce a balanced split.
Failure of C or D rejects it as a reusable local composition rule.  Failure of
E alone means any improvement is generic scalar rebalancing rather than a
writer-specific circuit property.  A discovery pass only freezes a scalar for
a genuinely new behavioral panel; it is not OOD, extraction, or removal
evidence by itself.

## Price

One checkpoint load; 48 opened prefixes; five writer directions; 31 fixed
lambdas; batched evaluations of the local MLP9 stage only; four leave-one-family
selection folds; no suffix execution, readout, behavioral logit, fitting,
gradient, parameter update, or physical downstream intervention.
