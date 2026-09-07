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

That construction holdout is now an honest parent-level null.  On 94 disjoint, jointly capable
temporal-v15/is-was-v15 rows, mask 26 remains extremely close to full rank16 (worst behavior or
signed-response gap `.00282`) and introduces no collateral beyond full.  Temporal full/mask26
behavior remains `.82078/.81896`.  But the complete old rank16 parent fails on the new is-was
constructions: full behavior is only `.06754`, signed response `.66534`, and response RSE `.39883`;
mask26 is correspondingly `.06710/.66816/.39258`.  Terminal `parent_construction_failure`.

Therefore mask26 is a stable compression of this parent, not yet a construction-general source
circuit.  The next split must compare complete native `MLP0/1/2/3/6` write patches with the failed
rank16 patch on the new family.  Success of the whole-module patch would localize the failure to the
learned response subspace; failure would show that the native source-site graph itself changes.

### Complete-module construction atlas: old readers replay, behavior does not

The v15 complete-source atlas is valid at 50/64 forwards and returns
`source_graph_construction_failure`.  Replacing the entire donor-minus-recipient output at all five
old source MLPs is much better than the rank-16 intervention in the old response coordinates:
is/was signed response rises from `.66534` to `.92807` and RSE falls from `.39883` to `.02922`.
It nevertheless recovers only `.39852` of the behavioral effect, far below the prospective `.75`
bar.  Gain 1.15 raises behavior to `.47127` and response to `1.06190`, so native gain also misses
the frozen no-worse prediction.  MLP1 is the largest singleton at `53.44%` of the five-site
behavioral effect, below the 60% localization bar; MLP3 has the largest old-response projection
(`.76039`) but only `.13743` behavior.

This is sharper than merely saying that the five-site source graph changed.  The old downstream
response measurements judge the all-five native patch nearly complete while task behavior says
that more than half the causal effect is missing.  The old finite reader family is therefore not a
sufficient observation map on the v15 construction.  Complete patches are also grossly nonselective
on the inherited control set (14/16 top-1 flips, median KL `1.751`), unlike the rank-16 parent's
2/16 flips.  The next atlas must treat answer margins, centered final residual/logit vectors, and
registered downstream readers as separate response blocks and scan every complete attention and
MLP module.  Only after the missing modules are localized should heads or within-module response
subspaces be split.

The subsequent all-layer atlas supplies valid singleton columns for all 36 native attention/MLP
modules.  The missing v15 behavior is attention-heavy and distributed: complete `attn:8`, `attn:9`,
and `attn:11` responses recover `.40565`, `.38479`, and `.27203` of behavior with 100% rowwise
direction; the largest MLP remains `mlp:1` at `.21296`.  No module reaches the prospective `.50`
localization bar.  Final-residual and behavioral singleton scores have Spearman `.648`, so the full
residual is a materially better observation block than the old reader-only response, though still
not interchangeable with behavior.

The same result's prefix interpretation is quarantined.  The input diagnostic replaces the entire
aligned embedding-output sequence; because base and donor prompts are the paired inputs, that arm
reconstructs the donor trajectory exactly.  Every input-inclusive prefix is consequently `1.0`
before any depth inference and cannot establish that layer 0 is sufficient.  This does not affect
the 36 independently executed module singletons, exact base-self/all-donor closures, or old-five
replay.  The next causal split is therefore all 27 heads in attention 8/9/11, measured both alone
and by leave-one-head-out inside each complete attention response.

The 27-head split is valid at exactly 60 forwards and identifies a sparse head-level core
inside those three live attention modules.  `L8H1` is the largest standalone contributor,
with behavioral signed projection `.40075` and perfect rowwise causal direction.  The other
material heads are `L11H3` (`.25460`), `L9H1` (`.16310`), and `L9H4` (`.15853`), again with
perfect direction.  The same four heads are conditionally necessary inside their respective
complete attention parents (`.40442`, `.25477`, `.18481`, and `.18082`).  Every all-nine-head
arm replays its complete-attention parent exactly and the three-layer base self-patch has zero
error, so the localization is not a head-layout artifact.

This result splits native modules but does not make `L8H1` a sufficient circuit.  Its `.401`
effect is localized relative to its layer, while the four material standalone effects together
span several serial layers and the full module atlas also contains material MLP writers.  The
scientific branch is therefore a frozen greedy composition over these four heads plus the live
complete MLP/module candidates.  That executed combination, rather than a sum of singleton
scores, will decide whether the v15 behavior admits a small cross-boundary circuit and which
pieces are selectively necessary.

The 17-candidate adaptive combination is valid at exactly 174 forwards but finds no selective
prefix on its registered greedy path.  The full bank reconstructs both behavior (`.95849/.96621`
on A1/A2) and the final residual (`.98098/.98789` signed projection), yet flips 29/32 P/C
controls with median KL `2.62725`.  The path first exceeds `.80` behavioral projection with five
pieces, but by then flips 24 controls.  Complete MLP1 is the main collateral transition: adding
it at step three raises A1 behavior from `.45713` to `.69867` while median control KL jumps from
`.01614` to `1.52048` and flips rise from one to 25.

The registered `no_small_selective_prefix` terminal is local to that greedy sequence, not a
global subset impossibility result.  `attn:15` and `L9H4` are individually zero-flip, low-KL
components, while `L8H1`, `L11H3`, and `L9H1` have only one or two singleton flips.  Because
composition can restore selectivity, the next exact test is the complete 32-subset lattice over
those five attention pieces.  Failure of that lattice to find a target-sufficient selective set
would isolate the remaining problem to task-conditioned splitting of the high-gain, high-collateral
complete MLP writes.

That selectivity conclusion is now retracted because the control intervention was not
token-position aligned.  Every A1/A2 target pair has equal base/donor token length and equal final
semantic position, so the target localization and composition measurements remain valid.  The
legacy P pairs do not (`6→7` tokens in the first sampled row), and the canonical C pairs also do
not (`15→13`).  The runners checked only equal **padded batch tensor shape**, then copied donor
responses to the base by absolute token index.  On P/C this mixes different semantic tokens and
can itself create the reported KL and top-one flips.  Therefore the greedy control-constrained
path and the attention lattice's `no_selective_attention_subset` terminal are invalid as
selectivity evidence; their target-only subset behavior remains usable.

A shared fail-closed contract now requires equal per-row token arrays and equal terminal semantic
positions before any absolute-index full-sequence patch can claim a control result.  The next
experiment must build capability-qualified, equal-length P and C controls and replay the complete
32-mask attention lattice.  Only if aligned controls reject the attention subsets should the route
move to within-MLP task/control subspace splitting.

That capability repair has now passed prospectively.  The replacement P panel pairs the two target
cue constructions at fixed tense, answer, noun, and token count; the replacement C panel uses an
unrelated equal-length nighttime-completion construction.  P/base, P/donor, C/base, and C/donor are
all 16/16 natively correct, and all 16 rows in each panel are jointly correct on both sides.  The
test used exactly two native forwards and opened no A1/A2 or causal outcomes.  This licenses the
unchanged 32-mask attention replay, with P and C collateral reported separately so one family
cannot hide failure of the other.

The licensed replay is complete and returns a valid null.  It exactly reproduced the old target
metrics within `2.92e-6` and executed all 32 masks at the registered 35-forward price.  Only the
full five-piece mask reaches the A1 target bar (`.80535`, with A2 `.86829`), but that mask changes
five of sixteen aligned P rows (median full-vocabulary KL `.11881`) and three of sixteen aligned C
rows (median KL `.01156`).  No mask is selective under the separate-panel rule.  This establishes
that the complete attention response slices are not a selective circuit, rather than merely that
the earlier control instrument was broken.  L8H1 is the largest single P collision (`.03106`
median KL and four flips); L9H1 and L11H3 each flip three P rows, while complete attention 15 alone
flips none.  The next test therefore splits the four material heads in response space with
cross-fitted task and P-nuisance spans while retaining attention 15 complete.

That linear split is also complete and informative.  A cross-fitted rank-eight task span keeps A1
at `.80380` but transfers only `.57337` on A2 and still flips four P rows.  Removing the opposite-
fold P nuisance span does what the complement loss requests: P/C flips fall to zero and P median KL
falls to `.00913`.  But target transfer simultaneously falls to `.57591/.40321` on A1/A2.  The
rank-one DIM and complemented-DIM arms show the same construction failure (`.72501/.43223` and
`.53597/.30410`).  The result rejects a shared linear feature projector as the selective object;
it does not reject optimization.  The next split uses the exact attention identity
`P'V' - PV = (P'-P)V + P(V'-V) + (P'-P)(V'-V)` to distinguish routing, value content, and their
interaction before defining any nonlinear DAS objective.

The exact attention operation factorial is now complete after preserving an invalid additive-
installation attempt.  The corrected absolute-clamp retry passes every mechanical tripwire at
exactly 17 forwards: raw factor closure is `1.14e-5`, base self is zero, and the full factor arm
replays the complete-head lattice within `2.53e-7` with identical categorical diagnostics.  No
homogeneous operation subset is selective.  Base-pattern-on-value-change carries nearly the whole
target effect (`.83077/.85314` A1/A2) but flips five P rows and one C row; pattern-only and
interaction-only remain nearly control-inert but reach only `.09366/.12631` and `.13699/.14621`.
Thus the useful computation is value transport, while its collateral is entangled across the four
heads.  The next exact split assigns the value factor independently at L8H1/L9H1/L9H4/L11H3 while
keeping pattern and interaction as global bits, rather than optimizing a homogeneous DAS axis.

That 64-arm head-specific value lattice is also a valid null.  Its full arm replays the parent to
`4.62e-7`, all categorical diagnostics match, and the exact 73-forward price holds.  No arm is
selective.  Every strictly zero-flip, low-KL arm contains no value factor and reaches at most
`.12692` A1.  L11H3 value is the strongest value singleton (`.37984/.36299` A1/A2) but flips three
P rows; L8H1 and L9H1 value each reach about `.308` A1 with two and three P flips, and L9H4 value
reaches `.202` with one P and one C flip.  Head allocation therefore does not disentangle the
target-dominant value computation.  The final exact-operation screen opens all 12 head×factor
components under deterministic target-first and target/control-constrained greedy paths, allowing
head-specific pattern/interaction terms to offset collateral before any learned subspace.

That 12-factor dual-greedy screen is now complete and valid.  It evaluated 143 unique scientific
arms in 152 forwards, with zero self error, `1.14e-5` factor closure, `4.62e-7` numeric parent replay,
and exact categorical replay.  Neither deterministic path finds a selective program.  The strongest
strictly zero-flip/low-KL visited arm reaches only `.13763` A1.  The target-first path reaches
`.84510` A1 but flips five P and two C rows; the full arm reaches `.80535/.86829` A1/A2 and flips
five P plus three C rows.  The terminal `no_selective_dual_greedy_program` is path-local: it closes
the discrete exact-factor allocation route, not every continuous projector.  The registered
successor is therefore a head-local, cross-fitted DAS projector with A1 transfer as a hard
feasibility constraint and P KL, Gaussian-noise sensitivity, and fold-projector stability used to
choose among feasible projectors while A2/C remain sealed.

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

The subsequent head-local target-feasible regularized DAS test resolves a different ambiguity and
requires a more precise verdict than its machine label `regularization_does_not_beat_dim`.  The
instrument is valid: absolute-clamp parent replay is within `7.15e-7`, factor closure within
`7.63e-6`, native/manual logits within `1.81e-5`, and the run used exactly 1,738 differentiable
forwards, 1,248 backward forwards, 480 updates, and 28,128 example evaluations.  Both parity fits
select the same rank-one, factor-SVD-initialized family at step four (`sigma=.05`, Jacobian weight
`.25`).  The learned directions are stable (minimum principal cosine `.8474`; maximum normalized
projector Frobenius distance `.5309`) and improve cross-fit A1 signed projection from matched DIM's
`.7250` to `.8719`, with perfect direction agreement.  Optimization therefore did find a
materially stronger target solution; this is not evidence that DAS optimization cannot outperform
difference in means.

What fails is selective construction transfer.  The learned projector reaches only `.6489` on the
sealed A2 construction, below the frozen `.75` floor, despite improving over DIM's `.4322`.  Its
cross-fit P median KL improves (`.00175` versus `.00315`), but one parity direction has mean P KL
`.01945` versus DIM's `.00678` and flips one row; the other direction improves to `.00107` versus
`.00201`.  The tiny advantage over the corresponding unregularized candidate in the selection
score (`.132217` versus `.132714`) also shows that Gaussian response noise and the local Jacobian
penalty did not materially change the geometry.  Thus the user hypothesis was partly confirmed:
the constrained optimizer is better at target extraction, while the present regularization and
single-construction objective do not identify a construction-general selective subspace.  This is
a construction-specific-geometry result, not a blanket DAS null.  Weight-tensor translation of
this direction remains diagnostic only; it cannot yet be promoted as an identified shared variable.

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
  MLP1/3/6; complete 32-mask composition and selectivity), followed by
  `temporal_iswas_rank16_source_mask26_construction_holdout_v1_result.json` (parent-level is-was
  construction null; mask remains close to full but full is not functional), followed by
  `temporal_iswas_v15_complete_source_mlp_write_atlas_v1_result.json` (valid source-graph/reader
  failure: old response coordinates replay at `.928` while behavior reaches only `.399`), followed
  by `temporal_iswas_v15_all_layer_complete_module_response_atlas_v1_result.json` (valid 36-module
  singleton atlas; top `attn:8/9/11`, while input-inclusive prefix inference is tautological and
  explicitly quarantined), followed by
  `temporal_iswas_v15_attention8_9_11_complete_head_atlas_v1_result.json` (valid exact 27-head
  singleton/conditional atlas; material core `L8H1`, `L9H1`, `L9H4`, and `L11H3`), followed by
  `temporal_iswas_v15_cross_boundary_adaptive_greedy_v1_result.json` (valid greedy-path
  target-composition evidence but invalid P/C selectivity instrument due unequal per-row token
  lengths), followed by `temporal_iswas_v15_low_collateral_attention_lattice_v1_result.json`
  (complete target-side five-piece lattice; selectivity terminal retracted for the same control
  alignment defect), followed by `temporal_iswas_v15_aligned_control_attention_lattice_v1_result.json`
  (valid aligned complete-head selectivity null),
  `temporal_iswas_v15_crossfit_head_response_task_p_complement_v1_result.json` (linear complement
  removes P but destroys construction transfer), and
  `temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1_retry1_result.json` (valid exact
  operation null: value transport is target-dominant but nonselective when shared across heads).
- Fully instrumented DAS regularization:
  `temporal_h3_das_family_crossvalidated_regularization_tournament_v2_result.json`, followed by
  `temporal_h3_das_nested_construction_adaptive_regularization_v1_result.json` (valid rejection of
  the current scalar-axis objective), followed by
  `temporal_iswas_v15_head_response_target_feasible_regularized_das_v1_result.json` (valid,
  fold-stable target improvement over DIM but failure of construction-general selectivity).

## Remaining gates

1. Test the frozen task-rank-four programs on a genuinely new capability-qualified lexical
   and construction bank, without refitting.
2. Capability-qualify a genuinely new cue construction, then fit a multi-environment causal-response
   object on complete constructions while keeping the new construction sealed.  Retain DIM, step
   zero, and exact-factor programs as controls.
3. If the multi-environment fixed projector remains construction-specific, replace it with an
   input-conditional/nonlinear causal-response object whose held-out blocks are complete constructions
   and downstream readers; do not merely widen the same coefficient grid.
4. Test joint composition when temporal and is–was commands are installed together.
5. Price an extracted executor only after these identification gates pass.
