# Folding the attention partition back to its input readers

The weight analysis supports Logan’s distinction between **shared output modules** and **shared input computations**. The existing projector/remainder split often reads overlapping value features. It does not yet identify two separate semantic algorithms.

At one block, concatenate the selected head outputs and write the saved direction as q, with head slice q_h. Let O_h be that head’s output matrix and w=Oq the saved residual writer. The exact head contributions to the two branches are

    O_P,h = w q_h^T,
    O_R,h = O_h - w q_h^T.

They sum to O_h. Each multiplies the SAME native attention pattern for that head, including both QK score factors, all four norm factors, position transforms and the causal mask. The output-space split changes neither QK1 nor QK2. The previous failed routing-only/value-only sufficiency test did not separately identify task-specific QK factors.

Fold the value projection and the first-layer value path into

    Vbar_h = [(1-lambda_h) V_local,h, lambda_h V_first,h].

This reads a 2,304-dimensional pair of normalized local and first-layer inputs. Lambda is the trained mixing coefficient; it need not lie between zero and one. The saved branch reads the scalar

    a_h = q_h^T Vbar_h.

The remainder reads the matrix `(O_h-w q_h^T) Vbar_h`. We asked whether a_h lies in the row space of that matrix. Row space means the collection of input linear functions accessible through a matrix’s outputs. Membership means the remainder has access to that input linear function; it does not mean it uses it identically.

## What the trained weights say

Across the 26 selected heads, 19 remainder writers retain all 128 head coordinates at the registered relative singular-value threshold 1e-6. For these heads, the saved scalar reader lies entirely in the remainder’s value-input row space: maximum relative projection residual 1.99e-15.

The seven blocks containing only one selected head instead have numerical remainder rank 127. Their squared input-reader overlaps are:

| Head | Squared overlap |
|---|---:|
| L3H0 | .098 |
| L6H1 | .210 |
| L7H8 | .146 |
| L10H5 | .168 |
| L14H8 | .061 |
| L15H1 | .071 |
| L16H8 | .076 |

These are geometric fractions of the scalar reader’s squared Euclidean norm in the specified two-input coordinates, not causal recovery or explained behavior. Using separately normalized q vectors changes overlaps by at most 2.6e-8. At the tighter rank threshold 1e-7, stored-q rounding makes L7H8 and L15H1 appear full rank; the normalized-q control distinguishes that numerical issue. No saved circuit direction was changed.

A synthetic routed evaluation using these trained matrices verifies that the two contributions sum to the original head output with relative error below 6.48e-16. The value reader agrees with the previously saved folded port within 5.78e-15 maximum absolute error. This is a CPU weight calculation, not a new native-model causal experiment.

## Why this matters for the proposed circuit split

The multi-head remainder includes compensating contributions across heads. Its per-head input overlap does **not** establish global redundancy: the full sum can cancel information, and the heads have different routing functions. Consequently, per-head row spaces are a useful starting point, not a full identification of the shared input-product space. Normalization and the coordinates in which overlap is measured remain explicit.

A more informative semantic split could indeed share a value feature while using different QK1/QK2 routing factors across heads. The exact weights above tell us which value readers are available. To establish that candidate, we must identify which routing factors distinguish the behaviors and test their swaps/removals in the live model. Naming the entire 3,314-coordinate remainder as the second circuit would skip that step.

The next user-directed experiment examines another source of specific readers: individual unembedding token vectors and a hierarchy of unembedding weight clusters, folded backward through MLP17 and MLP16. It tests the intervening attention path explicitly, and preserves all token-specific residual readers unless a fixed shared-reader test succeeds.

Evidence: [CPU receipt](../../CORRELATIVE_OV_PULLBACK_V1_RESULT.json), [audit implementation](../../audit_correlative_ov_pullback_v1.py), [existing folded scalar program](../../correlative_route_read_write_v1.py), [next experiment](../../UNEMBEDDING_BACKWARD_VIEWS_V1_PREREGISTRATION.md).
