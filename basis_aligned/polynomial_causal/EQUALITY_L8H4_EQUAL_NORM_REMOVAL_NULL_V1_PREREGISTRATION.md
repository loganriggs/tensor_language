# Equality L8H4 equal-norm removal null V1

## Status and question

This is a prospective control calculation over the already frozen 192-document
code-OOD authority. It does **not** add a fresh OOD panel. The target edge,
site, behavior masks, package, and V3 removal result are already fixed.

Question: is the behavioral damage from removing the exact L8H4 equality edge
specific to its residual direction, or would an arbitrary same-site direction
with the same deployed norm produce comparable damage?

## Frozen intervention

For each document, query position, and each of 16 seeds, draw an isotropic
Gaussian vector in the 1,152-dimensional L8H4 output-write space. Normalize it
to the exact BF16 deployed equality-term norm at that same document and query
position, cast it to the native attention-write dtype, and subtract it at the
same L8H4 write boundary. Positions with zero target norm receive a zero
control. No direction is selected using behavior.

Seeds are `2026091700` through `2026091715`, with the batch start added to make
the random stream independent across batches. The target arm is the exact V3
canonical removal. The native model, copy masks, two 96-document halves, and
all-noncopy collateral mask are inherited unchanged.

## Registered predictions

1. `pred_a_instrument_and_equal_norm`: all inputs are hash-bound and finite;
   each deployed random direction matches the target per-position norm within
   1% relative error; exact price and checkpoint identity hold.
2. `pred_b_target_removal_is_behaviorally_material`: target removal has
   positive summed NLL damage on `copy_positive`, all four copy subcells, and
   both frozen halves.
3. `pred_c_target_beats_equal_norm_directions`: target `copy_positive` damage
   exceeds at least 14/16 absolute random-control damages and is at least twice
   their median absolute damage.
4. `pred_d_target_is_selective`: target all-noncopy mean absolute change is at
   most `.01` nat and no larger than the median random-control collateral.
5. `pred_e_fixed_zero_fit_control_bank`: exactly 16 registered seeds are used;
   there are no fits, gradients, or parameter updates.

The result is a selective-removal confirmation only if all five pass. If the
instrument passes but any scientific prediction fails, the terminal is a valid
equal-norm null and the L8H4 edge remains a reversible/OOD/compositional
boundary without a certified selective-removal trait. A failure does not
invalidate its exact algebra or the earlier OOD installation result.

## Price

192 frozen documents, batch size inherited from the V3 authority, 18 full
trajectories per batch (native, target removal, and 16 controls), 864 forward
calls, one checkpoint load, zero fitting/backward/update calls, timeout 240 s.
