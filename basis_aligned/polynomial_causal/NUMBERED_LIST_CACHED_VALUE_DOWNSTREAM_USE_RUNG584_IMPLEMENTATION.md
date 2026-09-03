# Rung 584 implementation freeze for the R582 downstream-use experiment

R584 implements the already-frozen R582 scientific design.  It does not change the rows, sites, component order,
thresholds, split rules, or scientific nulls.

## Exact execution object

For each prompt, the runner performs one native and one R576-term-deleted prefix trajectory.  The deleted trajectory
subtracts exactly

`sum_(h in {3,7}) p8[h,q,k] WO8[h](lambda8 WV0[h] z0[k])`

at the final query after attention 8, using the prompt's frozen final-source position `k`.  Both trajectories then
recompute normally.  At MLP8/10/12/14 the runner records the separately RMS-normalized states and computes R582's
exact `C_l`, `Q_l`, and `C_l+Q_l` vectors from the live `Left`, `Right`, and `Down` weights.  A component intervention
runs the native prefix, subtracts the frozen vector from that MLP's final-query write, and recomputes the suffix.

FIT evaluates all twelve site/component candidates in the preregistered order.  The first candidate passing the
non-null R582 gates is provisional.  The two frozen active nulls are then evaluated only for that provisional
candidate and only on the factorial/surface successor/copy rows on which their paired action-gap comparison is
defined.  A failed null does not permit selection of a later candidate.  SELECT opens only if the provisional
candidate also beats both FIT nulls.  SELECT evaluates all three components at the selected site for interaction
accounting, but only the originally selected component can pass.  FINAL_TEST and OOD remain closed.

`pred_a` means all opened native replays, exact R576 deletions, and finite MLP response reconstructions meet `1e-10`.
`pred_b` means the fixed selected component passes action separation, necessity, copy preservation, source/surface,
conflict, and null gates on FIT and conditional SELECT.  `pred_c` means that same component passes these gates in all
three representations, supporting reuse.  The terminal claim requires all three.

## Evidence and price correction

Each raw row saves registered candidate logits and log-sum-exp for CE, plus the squared full-vocabulary logit-change
sum and vocabulary count for RMS.  It also saves source-deleted counterparts, component norms, exactness errors, and
null donor IDs.  This is sufficient for an independent CPU recomputation without serializing every 50,304-logit
vector for every arm.

R582's 530-forward ceiling conservatively priced both nulls on all 1,440 prompt types.  The executable nulls are
defined only on the four factorial/surface successor/copy cells, so R584 separately batches that frozen eligible
subset.  Its dry run reports both the conservative 530 ceiling and the lower literal executable ceiling.  The runner
must stay below both, use zero backwards and zero weight updates, and write a scientific null rather than crash when
no candidate passes or a live null fails.
