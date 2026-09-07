# Temporal / is–was shared graph with task-typed response programs

## Status

This is an identified two-task causal interface, not yet a fully extracted whole-model
program.  A single physical source graph of 46 native components drives eight measured
response sites.  Inside that graph, temporal `will`/`had` and copular `is`/`was` use
different four-dimensional response programs.  The programs transfer in both interchange
directions and are selective, but their worst 10%-source-noise behavioral recovery is
`0.79708`, just below the frozen `0.80` adoption bar.  The eight-dimensional pooled
executor remains the noise-robust release.

## Physical support and interface

The greedy source graph contains 46 components:

`L0H3, L0H8, MLP0, L1H1, L1H7, MLP1, L2H2, L2H3, L2H6, L2H7,
L2H8, MLP2, L3H0, L3H4, L3H5, L3H6, MLP3, L4H0, L4H1, L4H3,
L4H4, L4H5, L4H6, L4H7, MLP4, L5H0, L5H1, L5H6, L5H7, L5H8,
MLP5, L6H1, L6H3, MLP6, L7H0, L7H7, L7H8, MLP7, L8H1, MLP8,
L9H1, L9H4, L9H7, MLP9, L10H5, MLP10`.

The measured response sites are `MLP1`, `MLP3`, `MLP4`, `MLP6`, `L8H1`,
`L9H1`, `L9H4`, and `L11H3`.  Greedy deletion removed `L1H3` and `L3H7`.
Removing the last candidate, `L10H5`, reduced the minimum bidirectional behavioral
projection to `0.799723`, so 46 is the frozen greedy boundary under the exact `0.80`
criterion.  This is empirical minimality under the registered deletion order, not a
global minimum proof.

## Causal behavior

The pooled eight-dimensional response projector is the robust executor.  Across three
independent 10%-cue-source-noise seeds and both interchange directions its minimum
behavioral projection is `0.800048`, minimum coordinate projection is `0.930818`,
maximum direction gap is `0.008122`, and no control top-1 token flips.

Task-conditioned modes split that shared physical interface.  Rank three is insufficient
(`0.799066` temporal and `0.779499` is–was).  Rank four passes clean forward execution
(`0.801335`, `0.804071`) and reverse execution (`0.800677`, `0.800228`).  Installing the
other task's rank-four program instead gives only `0.274508` temporal and `0.379984`
is–was recovery.  This cross-task causal failure is the main evidence that the same native
modules contain distinct task programs rather than one generic response axis.

With 10% source noise, rank-four coordinate recovery remains almost exact: the minimum
mean signed projection is `0.988757`, the worst site residual is `0.043865`, the maximum
direction gap is `0.007841`, and controls have zero top-1 flips.  Behavioral recovery,
however, ranges down to `0.797080`.  Therefore rank four is licensed as a clean,
bidirectional task-conditioned program, but not as the robust executor.  This near-threshold
miss is treated as a real null; the threshold is not relaxed post hoc.

## What the weights say

Exact MLP quadratic weight contractions and complete attention OV contractions agree with
the causal task split.  Across the four MLP response sites, the largest temporal/is–was
quadratic principal cosine is below `0.176`.  Across the four attention sites, the largest
complete-OV overlap is `0.626`; no site has a shared weight-function direction above the
registered `0.80` bar.  Thus the tasks reuse native locations and the source graph, but the
weight functions used inside those locations are task typed.

This is precisely where tensor translation helps: for a response basis `Q`, an attention
mode maps to explicit value-input and residual-output factors through `W_V^T Q` and
`W_O Q`; an MLP output covector `a = W_down^T Q` maps to the gauge-invariant quadratic
reader `S(a) = 1/2[L^T diag(a)R + R^T diag(a)L]`.  Ranking upstream writers and downstream
readers by their contractions with these task-specific objects can group cross-module pieces
that implement the same variable and split native modules that implement different ones.

## Five-MLP rank-16 source execution

A complementary source-side route now gives a smaller exact interface at five MLP boundaries:
`MLP0`, `MLP1`, `MLP2`, `MLP3`, and `MLP6`.  A rank-16 output basis at each site, installed
with frozen gain `1.15`, is functional in both interchange directions.  On the reverse test,
temporal/is-was behavior is `.7575/.9174`, response projection is `1.0630/1.0997`, response
RSE is `.01463/.01454`, and matched controls have median/max KL `.00077/.00320` with no
top-1 flips.  Forward/reverse behavior differs by only `.00581/.00436`.

This interface has an exact native-weight translation.  For source MLP `s`, let `Q_s` be its
`1152 x 16` output basis and let `Down_s` be the native `1152 x 4608` output weight.  Then
`A_s = Down_s^T Q_s` is a `4608 x 16` hidden-to-interface map and the installed write is
exactly `(delta_hidden A_s) Q_s^T`.  All five maps have rank 16; fit/fresh factor closure is
`8.11e-14`, and an orthogonal gauge replay closes at `2.96e-15`.  The causal source is not
localized to a small static hidden-unit set: effective widths are 1,462–3,782 units and only
MLP6 places at least 25% of its energy in the top 10% of units.

Activation-conditioned ranking improves on weight magnitude but does not make the whole source
sparse.  An exhaustive five-site lattice found one stable within-module deletion: retaining only
the activation-ranked top half of MLP0 while keeping full hidden support at MLP1/2/3/6 is
functionally equivalent to the full rank-16 parent.  On a different construction family it gives
temporal/is-was behavior `.8640/.9234` versus `.8642/.9233` for full support, and response
projection `1.0807/1.0985` with RSE `.01361/.01448`.  Higher-order site interaction energy is
only `.00138`, so the broad support is distributed but nearly additive.

Absolute selectivity remains a parent-level caveat.  The transferred mask and the full parent
flip the same two of 16 controls; median KL is `.01220` for the mask and `.01249` for full, and
their centered control effects differ by only 1.87% of the full-parent effect.  Thus the MLP0
split adds no measured collateral, but it does not repair collateral already present in the
five-MLP intervention.

The transferred source-weight test now forms task-conditioned matrices
`Z_task = delta_hidden (Down^T Q)`, uses canonical-angle mean/contrast blocks to identify shared
versus task-specific physical weight directions, checks even/odd stability, and executes the
blocks and their complement through the complete model.  It is a valid, informative null.  All
five source sites have crossfit-stable mean/contrast blocks, own-task blocks strongly beat
cross-task blocks, and the mean+contrast union is functional: temporal/is-was behavior is
`.8527/.8744`, with signed causal response `1.0734/1.0870`.  It also reduces the parent's two
control flips to one.

The union is not yet full-equivalent.  Full rank-16 behavior is `.8642/.9233`, so the is-was gap
is `.04891`, above the frozen `.03` bar, even though the response-projection gap is only `.01182`.
The orthogonal complement produces only `.01181` is-was signed response but `.04693` behavior.
Thus a small physical residual is nonlinearly amplified downstream.  The registered parent terminal
is `probe_basis_only`, not an identified task-pair source circuit.

The subsequent full-module add-back screen resolves where that amplified residual lives.  No single
source MLP is sufficient: MLP3 is largest but recovers only `51.81%` of the missing is-was behavior;
MLP6, MLP1, and MLP2 recover `22.78%`, `17.08%`, and `9.37%`, while MLP0 is negligible and slightly
opposing.  MLP3's behavior gain (`.02534`) is materially larger than its signed-response gain
(`.00828`), confirming downstream nonlinear amplification.  Yet the five singleton behavior gains
sum to the full add-back within `.000406` on is-was and `.000558` on temporal, and no singleton adds
a control flip beyond the parent.  The terminal is therefore `distributed_additive_complement`.
This licenses a frozen greedy/complete five-site composition test; it does not yet license selecting
a minimal add-back set from singleton scores alone.

The complete 32-mask composition lattice passes all five registered predictions and selects mask 26:
full complement at `MLP1`, `MLP3`, and `MLP6`, with the mean+contrast union retained at `MLP0` and
`MLP2`.  It is the first eligible prefix of the frozen singleton order and the best of two eligible
three-site masks.  Temporal/is-was behavior is `.86308/.91890`, within `.00444` of full; signed
response is `1.08011/1.09614`, RSE is `.01358/.01463`, and its control report matches the full
parent's two flipped rows with slightly lower median KL (`.01207` versus `.01249`).  Removing any
selected site makes the arm ineligible.  Across all 32 masks the maximum deviation from singleton
additivity is only `.000768` in behavior and `.000251` in signed response.

The selectivity constraint changes the scientific interpretation.  `MLP3+MLP6` already meets the
strict `.015` target-equivalence tolerance, but flips a different control row not flipped by full.
Adding MLP1 restores the parent's control identity while improving is-was behavior.  MLP1 is
therefore a selectivity-routing component in this composed source program, not merely another large
linear response writer.  The mask remains a promoted screen until it transfers without reselection
to a capability-qualified new construction family.

## DAS interpretation

The constrained-DAS result is a target-mismatch and family-memorization warning, not proof
that optimization is intrinsically worse than difference in means (DIM).  Unregularized
constrained DAS had held-out full-vocabulary error `0.4485`; centered full-vocabulary KL
reduced it to `0.2445`, near DIM's `0.2385`, while tangent noise alone stayed at `0.4486`.
A better aligned objective reached `0.233426` on sealed A2, slightly better than DIM, but
traded away a registered target-sufficiency bar.  Complete-family selection then rejected
the learned rotations and retained the pooled step-zero/DIM-like estimator.

A fully instrumented four-arm tournament now gives positive but heterogeneous regularization
evidence.  It compared no regularization, tangent noise, KL, and noise+KL while exchanging whole
v8/v10 construction families.  The run counted 24 native calls and 1,208 differentiable reader
evaluations, had zero manual-reader closure error, and passed every authority and price check.
KL was selected and the two fold axes were stable (`|cos|=.8827`).  It improved the v8-to-v10
worst score from `.7261` to `.6968`, but worsened v10-to-v8 from `.3486` to `.4007`, so the frozen
both-fold claim failed.  Nevertheless, the refit reduced then-sealed v12 mean/worst loss from
`.9803/1.0095` to `.3979/.4803`, improving vocabulary behavior on both panels within every hard
target limit.  The correct terminal is `regularization_fold_heterogeneity`: optimization found a
transferable improvement, but one global KL/noise setting did not eliminate construction-specific
fitting.

The failure mode is under-observation.  A scalar answer/complement loss only constrains
the contractions it observes; the optimizer can rotate inside their joint nullspace and
specialize to the fitted construction.  Complement inertness is necessary but not sufficient.
KL helps because it observes the full output distribution.  Noise can select a flatter point
inside an already feasible set, but it does not add missing task or reader constraints.

Accordingly, the next DAS-worthy object is a multi-environment finite causal operator with
whole construction and downstream-reader blocks held out.  DIM must remain an eligible
checkpoint; target extraction is a hard feasibility constraint; complement and KL choose
among feasible checkpoints.  A learned rotation graduates only if it beats DIM without
refitting on sealed task families and predicts through the exact weight-derived readers.

The completed nested test now rejects the current rank-one scalar-axis objective.  It is fully
valid at 48 native plus 2,448 differentiable-reader calls and 305 updates.  V8 inner selection
chooses noise `.10` plus KL `1.0`, but the refit loses to matched-budget no-reg on outer v10
(`1.2052` versus `1.1555` worst loss).  V10 selects step zero for every configuration.  A global
noise+KL refit lowers the sealed-v15 aggregate loss from `1.3122/1.8406` mean/worst to
`.6927/1.2115`, but violates one hard L15 target bound by `.15642`.  This is positive evidence that
regularization can improve the observed complement distribution and simultaneous evidence that the
six scalar terms are non-identifying: the improvement can discard a required downstream effect.
The registered terminal is `inner_selection_mispredicts_construction`.  Further coefficient grids
are closed; the next optimization object must expose a finite causal-response operator.

## Evidence ledger

- Minimal support: `temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json`
  plus the hash-bound count correction, and
  `temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_v1_result.json`.
- Robust pooled executor:
  `temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1_result.json`.
- Task split and rank boundary:
  `temporal_iswas_rank46_task_typed_mode_causal_factorial_v1_result.json` and
  `temporal_iswas_rank46_task_mode_complete_rank_ladder_v1_result.json`.
- Reverse and noisy rank-four validation:
  `temporal_iswas_rank46_task_rank4_reverse_ood_v1_result.json` and
  `temporal_iswas_rank46_task_rank4_bidirectional_source_noise_v1_result.json`.
- Exact weight evidence: `temporal_iswas_mlp_quadratic_reader_overlap_v1_result.json`
  and `temporal_iswas_attention_ov_task_usage_v1_result.json`.
- Exact five-MLP source compilation and hidden support:
  `temporal_iswas_five_mlp_rank16_hidden_weight_compiler_v3_audit_result.json`,
  `temporal_iswas_rank16_hidden_complement_factorial_v1_result.json`,
  `temporal_iswas_rank16_hidden_mask30_construction_holdout_v1_result.json`, and
  `temporal_iswas_rank16_hidden_mask30_control_attribution_v1_result.json`.
- Exact task-pair source-weight grouping:
  `temporal_iswas_rank16_source_task_pair_weight_groups_v2_result.json` (valid functional null;
  five-site residual localization required), followed by
  `temporal_iswas_rank16_source_complement_site_localization_v1_result.json` (distributed, nearly
  additive complement; no single-site explanation), and
  `temporal_iswas_rank16_source_complement_composition_lattice_v1_result.json` (minimal mask 26:
  MLP1/3/6; complete 32-mask composition and selectivity).
- Fully instrumented DAS regularization:
  `temporal_h3_das_family_crossvalidated_regularization_tournament_v2_result.json`, followed by
  `temporal_h3_das_nested_construction_adaptive_regularization_v1_result.json` (valid rejection of
  the current scalar-axis objective).

## Remaining gates

1. Test the frozen task-rank-four programs on a genuinely new capability-qualified lexical
   and construction bank, without refitting.
2. Confirm frozen mask 26 (`MLP1/3/6` complement plus `MLP0/2` union) on a capability-qualified new
   construction family without reselection; then factor shared and contrast contributions inside
   the repaired source program.
3. Replace the rejected scalar-axis DAS loss with a finite causal-response operator whose held-out
   blocks are complete constructions and downstream readers; keep DIM/step zero and target retention
   as explicit controls, not post-hoc explanations.
4. Test joint composition when temporal and is–was commands are installed together.
5. Price an extracted executor only after these identification gates pass.
