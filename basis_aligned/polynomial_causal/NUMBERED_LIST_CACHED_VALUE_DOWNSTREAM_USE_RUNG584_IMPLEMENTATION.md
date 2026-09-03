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

## Prospective pre-outcome repair and literal null rule (2026-09-03 UTC)

No R584 model outcome has been run. Before execution, the implementation is tightened in four ways without changing
the rows, candidate order, scientific thresholds, or intervention arms.

First, every real arm must contain exactly every authority row in its opened split, and every null arm must contain
exactly every registered null-eligible recipient. Null donor IDs must equal the deterministic R582 map and must obey
the map's source, representation, condition, group/action constraints. A missing row, group, or donor is an integrity
error rather than a smaller sample.

Second, results, raw evidence, and dry runs must be finite standard JSON. Undefined ratios are saved as literal
`null`, with `passed: false` and an explicit reason; artifacts are encoded with `allow_nan=False`. In particular,
zero or nonpositive denominators never become `Infinity` or `-Infinity`.

Third, every length batch locally compares the manual layer-8 attention replay with the native attention computation,
and every capture row saves that replay error. Every row also saves the C/Q reconstruction error separately for each
site, rather than only the maximum. The native-versus-source-deleted squared full-vocabulary logit difference and
vocabulary count are saved so its RMS is independently recomputable. This repeats local attention arithmetic inside
an existing trajectory but does not add a whole-model forward; the 510-forward executable ceiling remains unchanged.

Fourth, every capture and intervention record saves the token IDs, source/query coordinates, literal source value,
answer/candidate IDs, and explicit arm/site/component identity. The result names the nested zero-outcome dry-run
object `execution_plan`, distinguishes its planned zero calls from observed top-level calls, and hashes the exact
runner, owner tests, adversarial tests, generic result-contract helper, frozen rows/documents, deterministic null
maps, dry run, and checkpoint. The generic result contract checks the final row/group census, split closure, field
types, finite JSON, forward/backward/weight-update envelope, and provenance before publication.

The previously ambiguous R582 null sentence is frozen to its literal conservative reading. For each
`source_level x surface x null`, let `L_real(r)` be the candidate's already-computed action-gap 95% lower bootstrap
bound for representation `r`, using the candidate's original bootstrap cell ID and draws. Let `L_null(r)` be the
corresponding null bound. The comparison passes only when

`min_r L_real(r) > max_r L_null(r)`.

The real bounds are reused; they are not redrawn with a null-specific cell ID. Activity is checked separately in
each `representation x source_level x surface` cell as

`median(null intervention norm) / median(real intervention norm)`,

and must remain in `[0.8, 1.25]`. All three representation activity cells and the conservative cross-representation
bound must pass for that source/surface/null. A zero real median produces `null` plus reason and fails. These rules are
fixed before any model call and may not be relaxed to rescue an outcome.
