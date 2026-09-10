# Joint QK spaces before positional transforms

Controlled representation comparison after the valid V1 result. Same FITfirst8 A1/C, EVALlast8 A1/A2/C, all26heads, query/key spans and1e-6relative SVD tolerance. Same28forwards224seq and A/B/C bars as V1. No new independent holdout or threshold adjustment. V1 result remains valid for its post-rotary feature ports.

Change: map actual native q1/q2/k1/k2 back through the actual rounded rotary matrix. Inverse uses division by cos^2+sin^2; no orthogonality assumption. Fit and edit joint products in those pre-position coordinates, then apply R_pos to both axes of every joint feature before scoring. Both QK factors and native head-normalization responses remain. Query/frame coordinate change is thus removed from the proposed fixed reader identity, while routing retains actual relative positions.

pred_a retains V1 native reconstruction/noop/full-reference/coordinate/basis/capability gates, and CPU inverse/product transport relative error<=1e-10. pred_b own-subspace conditional full-logit routing error<=.20,projection recovery>=.80 all3panels. pred_c other-subspace effect norm<=.20 all3panels,reference norm>1e-4. Null insufficient or overlapping task spaces persists. Both sources of mixed terms retained. Product-port edits need not be native vector/weight edits.

Price unchanged native forwards; added inverse128-vector position transforms and128x128 joint-product transport. Largest basis48columns, per tensor<=256MiB. Save V2 bases separately. Numerical inverse acts on captured native rounded vectors; it does not claim exact recovery of pre-rounding internal floats.
