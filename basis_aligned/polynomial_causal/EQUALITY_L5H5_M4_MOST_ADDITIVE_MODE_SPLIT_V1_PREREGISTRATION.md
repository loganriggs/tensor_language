# Equality L5H5 M4 most-additive contracted-mode split V1

The float64-certified contracted-mode graph retains rank 256 and passes OOD
score prediction, causal use, and selectivity, but its arbitrary alternating
128/128 split has `.49047` component-relative score composition error.  This
experiment tests the briefing's most-additive split rather than interpreting an
arbitrary partition as a general composition failure.

The frozen rank-256 singular ordering and all weights are inherited.  On the
same natural selection role, evaluate nested partitions with prefix child rank
`8,16,32,64,128` and the complementary modes through 255 as the remainder.
For each partition, measure the separately evaluated score effects of child and
remainder relative to the rank-0/bias-only background.  A candidate is eligible
only if child and remainder effect norms are respectively at least `.50` and
`.10` of the joint rank-256 effect norm.  Select the eligible partition with
minimum component-relative additive-prediction error, breaking ties by smaller
child rank.  If none is eligible, select the minimum-error partition and mark
selection unqualified.

Frozen gates:

1. The inherited float64 bridge and native-product exactness gates pass.
2. Natural selected score composition error is at most `.10`.
3. Without reselection, code score composition error is at most `.10` overall
   and `.15` in both halves.
4. At the NLL interface, separately run bias-only, child, remainder, and joint
   donor arms.  Their additive prediction of the joint effect has at most `.10`
   relative L2 error on copy-positive tokens, `.15` in each half, and `.15` in
   each registered copy subtype.
5. The joint donor retains `[.80,1.05]` copy-positive recovery and at most `.01`
   nat noncopy damage; both child and remainder donor terms are nonzero.

No behavior labels or code rows select the partition.  This does not relax the
rank-64 compactness null or claim product-cost compression.  Failure after a
lawful inherited bridge is a valid most-additive-split null.

Price: one checkpoint load, one natural partition pass, one code score pass,
and six code behavior arms; no gradients, parameter updates, fits, or new text.
