# The correlative circuit needs a routing–value product

The new test finds that value content carries most of the circuit's cue effect,
but the full operation requires routing as well. Neither routing-only nor
value-only swaps reproduce the complete circuit under the registered criteria.
The operation is an explicit product; simplifying it to independent contributions
loses a substantial interaction before the downstream model even runs.

This extends the original handoff's weight-folding direction. It does not revive
is/was or the failed long-sentence grammatical claim. We used the existing short
sentences where the model had answered both cue variants correctly, kept the
saved circuit fixed, and checked that capability again.

## The operation and experiment

The saved interface comprises26 attention heads grouped into14 blocks, one
scalar per block. A scalar is a single number read from that block's selected
heads. Folding the learned direction through the native value and output
matrices gives

    s = sum over heads and source positions of P * U,
    residual edit = w * change_in_s.

P is the native attention score: a product of two normalized query–key dot
products, with the actual position rotation and causal mask. U is the scalar
value read, including both the local contextual value and the first-layer value
with the model's signed mixing coefficient. w is the fixed output vector that
writes the scalar change into the residual stream.

We captured P and U from a donor sentence and installed three operations in
the recipient: donor routing with live recipient values; donor values with live
recipient routing; and both donor operands together. Each block recomputed its
current recipient inputs after earlier edits. Thus this is a live intervention,
not a calculation using a stale unedited recipient.

The four panels were the existing16-group bare and report frames, a reporter
change preserving the answer, and a disjoint either/not task with or/but answers.
They are reused test data, not a new out-of-distribution sample. No fitting,
head selection, rank adjustment or gain tuning occurred.

All64 pairs passed both-endpoint native correctness. The new joint executor
passed its full-vocabulary bridge to the existing scalar executor, and the
native baseline bridge passed. The managed run used28 forwards/448 sequences,
1.654 seconds of recorded executor time, at14:07:32–14:07:36 UTC.

## Results

Recovery means the fraction of the native cue-induced answer-margin change
reproduced by the swap. Vocabulary error is the Euclidean error in the centered
change across all50304 output scores, divided by the complete swap's change.
It is not an error rate over predicted tokens.

| Swap | Bare-frame recovery | Report-frame recovery | Vocabulary error, bare/report |
|---|---:|---:|---:|
| Both operands | .969 | .899 | 0 / 0 by definition |
| Routing only | .159 | .119 | .829 / .860 |
| Values only | .853 | .780 | .209 / .221 |

The sufficiency requirement was recovery>=.8 and vocabulary error<=.15 in
both target frames, plus small control effects. Both partial operations fail.
Values carry much more of the target effect, but that does not make them a
sufficient extracted circuit. Control movements were small for every arm,
at most.0248 of the reference native cue separation, below the.23 threshold.

The combined endpoint effect also differs from the sum of the singleton
effects. Its relative interaction was.348 in the bare frame and.306 in the
report frame, above the.10 additive-effect threshold. Paired bootstrap intervals
over the16 authored groups were[.316,.379] and[.286,.327]. These describe this
finite panel; they are not population or model-training guarantees.

## What the product calculation explains

At a fixed native recipient state, define delta_P=P_donor-P_base and similarly
delta_U. Ordinary multiplication gives the exact identity

    delta_s = sum(delta_P * U_base)
            + sum(P_base * delta_U)
            + sum(delta_P * delta_U).

The first term changes routing, the second changes values, and the third is
their interaction. The last term is required even though the operation is
bilinear. “Bilinear” means linear in either operand while the other is fixed;
it does not mean additive when both change.

The CPU audit multiplied each term by its block's actual output writer and
stacked the resulting block writes. In that norm, omitting the cross term loses
.384 of the full native write change in the bare frame and.302 in the report
frame. The product identity agrees to about1.7e−16 relative error on the targets.
These native-state terms do not by themselves predict multi-block intervention
effects, because each partial intervention follows a different live trajectory.
Nor is a norm across block writes the same as a final-vocabulary norm.

This establishes two distinct interaction requirements: the local product has
a substantial cross term, and the separately executed endpoint effects fail
addition. We have not attributed a percentage of the endpoint interaction to
the local cross term; that would require a separate causal comparison.

## What remains unresolved

We now have explicit operand ports for a causally effective computation, plus
evidence against removing either operand. We still need to explain and execute
the routing and value producers independently, identify their consumers, and
show selective removal and reuse. The scalar weights alone do not supply that
explanation. Neither of the failed operand-only hypotheses will be rescued by
a new gain or a larger fitted direction.

All545902902 native parameters and the existing76032 folded reader/writer
coefficients remain charged. No structural saving or independent extraction
was achieved. The useful consequence is that a proposed simpler program must
predict the coupled routing–value computation, rather than assuming its two
inputs contribute independently.

Receipts in the parent directory: `CORRELATIVE_ROUTE_VALUE_V1_RESULT.json`,
`CORRELATIVE_ROUTE_VALUE_AUDIT_V1_RESULT.json`, and the preregistration
`CORRELATIVE_ROUTE_VALUE_V1_PREREGISTRATION.md`.
