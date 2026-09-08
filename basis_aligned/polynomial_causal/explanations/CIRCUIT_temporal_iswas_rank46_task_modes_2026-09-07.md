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

The first exact translation of those four learned head directions into model weights is now a
valid diagnostic.  For a head-local direction $u_h$, the residual-stream write is computed exactly
as $w_h=W_{O,h}u_h$.  Candidate downstream readers are then ranked by normalized contractions of
$w_h$ with each attention query/key/value matrix (including the two query/key halves used by this
model) and each MLP left/right matrix.  Conversely, $W_{V,h}^{\mathsf T}u_h$ is pulled backward and
contracted with earlier attention output and MLP-down matrices to rank possible value writers.
This opens no activation or intervention outcome; it asks whether the already selected causal
coordinate is compatible with fixed weight interfaces.

The directions are stable across the two fitted folds (`|cos|=.9368-.9923`).  For every source
head and both folds, all five inspected interfaces of `L15H5` (`q`, `k`, `q2`, `k2`, and `v`) are
in the source's top ten downstream interfaces, and `L15H5` is at the `1.0` module-reader
percentile.  The backward value pullback also recovers the measured serial order: `L8H1` ranks at
percentiles `1.0/.9888` as a writer into `L9H1/L9H4`, while `L9H1/L9H4` rank at
`.9908/.9725` into `L11H3`, identically across folds.  The immutable first artifact was labelled
invalid because its exact $W_O$ embedding check used an absolute `1e-7` bar and observed
`2.3842e-7`.  A preregistered zero-model audit divided that error by the smallest write norm
(`9.2426`), obtaining `2.5796e-8`, below eight float32 epsilons (`9.5367e-7`); all hashes,
rankings, prices, and other predictions were unchanged.  This supports the concrete interface
chain `L8H1 -> {L9H1,L9H4} -> L11H3 -> L15H5`, but only at diagnostic weight-compatibility tier.
Fresh-construction causal transfer is still required before calling these weights readers and
writers of one identified reusable variable.

A subsequent static causal-path audit adds an important scope restriction.  Every fitted and
evaluated projector above uses a complete attention-15 donor clamp.  Inside the shared manual
intervention, the model first computes layer-15 attention, but the pre-`c_proj` hook then replaces
all nine computed head outputs with fixed donor-cache outputs.  Therefore the derivative from the
upstream projected writes through `L15H5` Q/K/V into its head output is exactly zero in these DAS
runs.  Upstream effects can still travel through the layer-15 residual skip into MLP15 and onward to
layers 16-17, so the projector's causal effect relative to an attention-15 background is not
invalidated.  What is not established is the tempting native edge from the exact weight atlas:
`L15H5` remains a reader hypothesis, not a causally identified reader.  Complete attention-15 alone
accounts for only `.12692/.12156` A1/A2 signed projection, so the background is weak but nontrivial.
After construction transfer is scored, the required next localization is the exact 2x2 factorial
`upstream projector on/off x complete attention-15 clamp on/off`; only an upstream-only effect through
live attention licenses Q/K/V interface reset/rescue.

The multi-construction rank-one test is a valid fixed-family null.  No initialization produces a
target-feasible selected checkpoint in both held lexical parities.  Factor-SVD misses one fold at
`.69925` minimum target projection while its other fold passes at `.75463` but flips two P rows;
joint-construction DIM misses one fold at `.70816`, and the frozen A1 initialization misses at
`.72091`.  The lexicographically selected frozen-A1 family is stable across folds (minimum line
cosine `.81967`), but only one fold is feasible and that fold retains one P flip.  Pooling the two
held folds would misleadingly look successful (`.80699/.81765` A1/A2), which confirms why the
registered per-environment, per-fold licensing rule must precede aggregate reporting.

On sealed v16, the selected projector reaches `.64172` A1 and `.49549` A2, so it also fails the
registered `.65` projection floor separately by construction.  V16 P has zero top-one flips, but
its mean/max KL are `.08436/.21270`; the flip-only registered prediction passes and does not imply
distributional selectivity.  The terminal is therefore
`fixed_projector_infeasible_on_observed_constructions`.  The preregistered attention-15 dependency
factorial is ineligible at this point.  The next branch is the mathematical falsifier: fit A1-only
and A2-only rank-one oracle axes under the same crossfit, measure their projective angles and
within-construction stability, then causally execute their exact sign-aligned bisectors.  This
distinguishes joint-optimizer failure from construction-conditioned rank-one geometry before any
input-conditioned mixture or reader reset is opened.

That falsifier validly selects construction-conditioned coordinates rather than optimizer failure.
Every construction-specific fit passes its own held target bar, with minimum within-construction
fold cosine `.86246` for A1 and `.99881` for A2.  Yet the A1-versus-A2 axes at the same head have
only `.40859-.67064` absolute cosine, and each own oracle beats cross-use by the registered `.08`
in both held parities.  The analytic per-head bisector reaches both v15 target bars in both folds
(`.76372-.93010`), but causes two P flips in parity 0 and does not beat the failed joint fit in both
folds.  Under the outcome-held v16 intervention it reaches only `.72390/.52098` on parity 0 and
`.59807/.45904` on parity 1 for A1/A2, with large parity-0 P mean KL `.18406`.  Thus geometric
overlap is not a behavioral guarantee through the nonlinear suffix.

A naïve construction router is already insufficient under the registered controls: both own
experts cause the same two parity-0 P flips, so choosing between them cannot produce a zero-flip P
arm.  The next diagnostic asks whether their different 128-dimensional head coordinates converge
after exact weight maps.  For each construction axis it computes the residual write through the
head's `W_O`, then compares the two response vectors through every later Q/K/Q2/K2/V or MLP
Left/Right reader, and compares their `W_V^T` pullbacks.  Shared residual or reader responses would
support a common downstream variable with construction-specific encodings; failure would keep the
linear interfaces construction-specific and require a routed or nonlinear selective object.  This
is a zero-forward weight screen, not causal reader identification, and the attention-15 bypass
restriction remains in force.

The weight-convergence screen returns `reader_equivalent_distinct_writes`.  Only L11H3's A1/A2
residual writes clear `.75` cosine in both folds; L8H1 is `.472/.512`, L9H1 `.569/.618`, and L9H4
`.721/.698`.  Value pullbacks are likewise shared only at L11H3 (`.804/.828`), so neither the
residual writes nor their upstream value covectors collapse to one common direction.  In contrast,
all five L15H5 interfaces are top-ten joint readers of all four sources in both folds.  Their
response alignments range roughly `.766-.966`, and each source's best L15H5 joint-score percentile
is `.997-1.0`.  Thus construction-specific encodings are linearly reader-equivalent at the exact
L15H5 weight interface even though they are not a common upstream vector.

This strengthens the L15H5 reader hypothesis while leaving its causal status unchanged: the
complete attention-15 clamp bypassed those Q/K/V computations in every behavioral run.  The next
causal discriminator therefore crosses each stable construction oracle on/off with complete
attention-15 on/off, without refitting.  Only retained own-construction transfer in the
projector-on/live-attention arm can open an L15H5-specific reset/rescue.  P controls remain explicit;
the existing parity-0 flips mean this dependency test localizes the reader path but cannot by itself
promote a selective full circuit.

The live-attention dependency factorial is now valid and returns `live_attention_route_candidate`.
All six registered predictions pass.  With native layer-15 attention, the A1 oracle preserves
signed target projections `.774/.747` across held parities and the A2 oracle preserves
`.750/.717`; every own-target direction fraction is `1.0`.  These are respectively at least 89%
and 91% of the matching fixed-attention effects.  The rowwise interaction is small relative to the
full response, and the live-attention arms have no worse P/C flips or mean KL than their fixed
backgrounds.  The reused v16 intervention also passes its registered retention and direction gates.

This establishes a native live-attention route for the fixed construction-specific upstream
coordinates, but it does not yet establish that layer 15, L15H5, or any Q/K/V factor mediates that
route: the upstream effect may still travel through the residual skip.  A hash-bound zero-model
gate therefore admitted the preregistered complete layer-15 module plus nine-singleton-head
reset/rescue atlas.  It will measure whole-module mediation before ranking heads, and will license a
greedy head union only if the sum of singleton reset losses approximates the module reset loss.

The admitted head/module atlas returns the valid terminal `attention15_bypass`.  Complete layer-15
attention reset loss mediates only `.057-.081` of the upstream live effect across the four
expert/parity own-target cells, below the registered `.10` bar.  L15H5 is nevertheless the top
reset and rescue singleton in all four cells and accounts for most of that small branch
(`.050-.072`).  Singleton reset losses reproduce the complete-module reset loss almost exactly
(cosine above `.9999996`, relative L2 below `.005`), and L15H5 controls are clean.  This licenses
neither a greedy union nor Q/K/V splitting because the complete attention module itself failed the
mediation gate.  On reused v16, L15H5 reset projection is `.031`, `-.015`, `.050`, and `-.030`, so
the small branch is not OOD-stable.  The dominant `.919-.943` effect remains after resetting the
whole attention-15 response and must be localized through carried residual state and downstream
module writes.

The complete layer-12--17 attention/MLP write atlas returns the valid terminal
`direct_residual_carry`.  Installing all twelve coherent upstream-off writes into the
upstream-on execution removes only `.141-.207` of the own-target effect, and installing all twelve
upstream-on writes into the upstream-off execution rescues only `.127-.210`; direction fractions
are `1.0`, so the small effect is aligned but insufficient.  No singleton module is stable above
the registered `.10` reset bar in all four expert/parity cells.  MLP13 is the top reset singleton
everywhere (`.082-.117`), while MLP17 contributes a large opposing effect (`-.037` to `-.130`).
The singleton sum is directionally close to the joint write bank (cosine `.984-.997`) but misses
the relative-L2 composition bar in two cells (`.498` and `.270`).  Exact rowwise closure and
self-clamp replay are zero, native replay error is `1.43e-5`, and the price is exactly 120 forwards.

This rejects a greedy union of downstream module writes and sharpens the circuit object: most of
the construction-specific upstream effect travels in the residual state around those module
writes, while small aligned and opposing write corrections modulate it.  The next prospective
atlas patches the complete residual state at the layer-12 entry and after each attention/MLP update
through layer 17.  It will locate the earliest boundary after which reset/rescue stays strong,
while leaving the embedding skip and recurrent attention state native; failure at an early boundary
therefore measures downstream reconstruction through those alternate state paths rather than being
silently pooled away.

The ordered residual-state atlas returns `nonselective_residual_lockin`.  Reset and rescue are
exactly `1.0` in every own-target cell at all thirteen boundaries, from `entry12` through
`post_mlp17`, with zero factorial and self-replay error.  This proves that the complete entry-12
residual tensor is sufficient and necessary for the measured upstream route and that the native
recurrent attention state adds no independent effect here.  It does not localize a later consumer:
because the upstream intervention changes only c-projected head output and leaves recurrent `v1`
unchanged, setting the complete residual state equal makes the deterministic suffix executions
identical.  The flat curve is therefore a graph-level state-interface result, not evidence that
every later module separately mediates the behavior.

Selectivity still fails exactly where the oracle fits failed it.  A1 parity 0 has one P flip with
reset/rescue mean KL `.0191/.0205`; A2 parity 0 has two flips and KL `.0463/.0511`.  Both parity-1
directions remain zero-flip and low-KL.  The next useful object is consequently not another complete
state swap, but a cross-fitted target/control causal-response basis *within the entry-12 state*.
That basis must predict both constructions and remove the P collision under projected state
interchange before weight-tensor translation can nominate its actual downstream readers.

The entry-12 shared-basis screen returns `p_complement_destroys_target`.  The fixed two-construction
DIM union is causally sufficient on every held expert/parity cell (`.795-.881` signed projection,
direction fraction `1.0`), whereas pooled rank-one DIM fails both A1 cells (`.674/.588`) even though
it passes A2 (`.956/.847`).  Thus the complete state contains a stable two-coordinate linear target
union, not one shared scalar.

The exact training-P complement removes every P/C top-one flip and lowers P mean KL below
`.00039`, but preserves only `.023-.058` target projection.  The training P response span has rank
56/58 at the frozen `1e-6` threshold.  This reproduces the earlier head-response result at the
causally sufficient state interface: the complement objective works exactly as specified, while
the target and P effects overlap in the linear causal-response space.  More optimization of the
same orthogonal-complement loss cannot recover the removed target component.  The licensed next
object is a prospective input-conditioned finite mixture: a cross-fitted router reads only native
entry-12 state, chooses the A1 coordinate, A2 coordinate, or off, and installs the corresponding
member of the sufficient two-coordinate union.  This tests conditional selectivity rather than
another rank or penalty sweep.

The first finite router returns `router_identification_failure`.  Its gold-label program composes
the two coordinates successfully and exactly preserves controls, but the learned native-state
router reaches only `.729/.667` held macro accuracy and predicts a target branch for 7/16 and 8/16
off rows.  Consequently learned target projection falls to `.655/.772` in parity 0 and
`.795/.646` in parity 1, and parity-0 P still flips once.

This is not evidence against conditional routing in general.  It exposes a structural information
error in the registered router input: every aligned P control has exactly the same base text as its
paired A1 target, hence their upstream-off native entry-12 states are mathematically identical.
No classifier on that state can distinguish whether the requested source transformation changes
tense or merely paraphrases the temporal cue.  The circuit operator receives both background and
source; the next prospective router must therefore read the proposed source-minus-background
entry-12 write (with no held label or output margin), then select A1, A2, or off.  This changes the
information object rather than tuning nearest-centroid geometry.

The source-delta nearest-centroid router is also a valid identification null.  It sends every held
row to `off` in both parities: macro accuracy is exactly `1/3`, target projection is zero, and P/C
are therefore perfectly preserved.  The gold mixture still reaches the union's `.795-.881` target
projection with zero control effect.  This separates two facts cleanly: the causal coordinates and
their finite composition work, while an unsupervised centroid rule on raw 2304-dimensional source
deltas does not identify their branch across lexical parity.

The next router test is supervised but fail-closed against memorization.  A class-balanced linear
ridge router on the same source-only features will choose its penalty by leave-one-group-out folds
entirely inside the opposite-parity training split, with P/C false positives prioritized before
macro accuracy; the held parity remains sealed.  This uses the correct branch-classification target
rather than the destructive complement objective.  Failure would license nonlinear interaction
features or a token-level symbolic router, not another ridge/noise sweep.

The nested regularized source router also returns `router_identification_failure`.  Its selected
penalties (`10.0` and `.01`) both achieve perfect leave-one-group-out accuracy with zero control
false positives *inside* the training parity, yet the refit predicts `off` for every row in the
opposite parity.  Held macro accuracy is `1/3` and target projection is zero.  This is a concrete
memorization/distribution-shift failure despite strong L2 regularization and nested group CV.

The missing invariance is exact: even group numbers are `present_to_past` and odd group numbers are
`past_to_present`.  The parity holdout therefore reverses the signed source delta.  A linear router
on raw deltas can fit lexical groups within one direction but is not gauge-invariant to the reverse
transformation.  The next bounded router uses only the six scalar entries of semantic-position and
prefix-averaged Gram matrices of the two proposed expert writes (two squared norms and one inner
product at each scope).  These quadratic features are invariant to simultaneous sign reversal and
directly test the interaction structure; failure closes this finite response-signature route.

The six-scalar Gram router partially succeeds but fails its strict gate.  One held direction is
perfect; the other reaches macro accuracy `.8958`, routes three off rows to A1, and misroutes one
A2 row to A1.  Both target panels nevertheless pass (`.795-.847`) and retain at least 90% of the
ungated union.  The remaining parity-0 P leakage is one flip at mean KL `.00512`.  Thus quadratic
sign invariance repairs direction reversal and target composition, but coarse response magnitudes
do not identify the off branch exactly.

Per the preregistered kill rule, the response-signature route now closes rather than receiving a
threshold tweak.  The next object is an explicit finite token-pair router: derive the unordered
aligned base/source token-difference signature, learn only the A1 and A2 signatures from opposite
parity, and default every unknown signature to off.  This is a transparent input-level gate with no
behavioral outcome access.  A v15 pass is only a screen and must face unseen cue-pair OOD before
identification, but it establishes whether the already-sufficient state coordinates can form a
selective executable program with an explicit router.

That explicit program passes all five registered predictions and returns
`token_pair_router_candidate`.  Each parity learns the same two unordered one-pair signatures,
maps them to A1/A2, and defaults everything else off.  Held confusion is perfect in both directions.
The routed state union projects `.795-.881` on all four target cells with direction fraction `1.0`,
has exactly zero P/C KL and flips, and matches gold-router logits bit-for-bit.  Its literal learned
gate contains two signature entries; the causal state portion remains the cross-fitted rank-two
union.

This is the first selective executable v15 program in the construction-oracle branch, but it is a
screen rather than identification.  Its finite token lookup is expected to default unseen cue pairs
off, so fresh-cue OOD must be audited without relabeling or adding signatures.  In parallel, the
successful selective state union now licenses the user's requested exact-weight diagnostic: map
both entry12 coordinates through downstream Q/K/V and MLP input tensors and the final unembedding,
then causally confirm only readers whose complete-module path has enough effect.  Static weight
alignment remains nomination, not proof, by the earlier L15H5 counterexample.

The frozen tied-embedding semantic router fails before OOD transfer.  Both opposite-parity v15
fits identify every A1 and A2 target, but each routes all eight held P rows to A2; held v15 macro
accuracy is therefore `.8333` with eight control false positives per fold.  On capability-qualified
v16, both frozen fits route every A1, A2, and P row to `off`, for macro accuracy `1/3` and zero
target coverage.  The two folds agree exactly, so instability is not the explanation.  A fixed
unordered midpoint plus squared-difference representation of tied token embeddings does not expose
the operational task distinction in a usable cosine-centroid geometry.

This null rules out the simple lexical-weight route, not semantic routing in general.  The next
bounded test returns to the already causally validated contextual object: the six sign-invariant
Gram scalars of the proposed A1/A2 entry-12 responses.  It fits one centroid rule across both v15
directions, checks leave-one-group-out selectivity, and evaluates sealed v16 under both independently
cross-fitted oracle bases.  This removes the earlier direction/parity confound without adding an
optimizer; disagreement between oracle folds or failure on v16 closes this contextual Gram route.

Pooling both tense directions does not rescue the six-scalar contextual Gram router.  Exact
whole-group leave-one-out on v15 gives confusion `[[16,0,0],[1,15,0],[3,0,29]]`, macro accuracy
`.9479`, and three control false positives.  Capability-qualified v16 exposes a stronger failure:
the two oracle folds achieve macro accuracy only `.1458/.125`, almost completely exchange A1 and
A2, leak `9/11` P rows, and disagree.  The earlier parity split was therefore not the sole cause;
coarse response energies and inner products do not stably identify construction semantics.

The registered successor changes the measured object once, from coarse moments to a fixed
fourteen-scalar projective response-shape spectrum (singular-energy concentration, participation,
semantic-position concentration, roughness, and cross-expert geometry).  It keeps whole-group v15
validation, both frozen v16 oracle folds, and no optimizer.  Its preregistered OOD failure closes
contextual centroid feature engineering rather than opening further variants.

The projective response-shape successor also fails its strict identification gate.  It separates
all 32 v15 target rows, but sends five held controls to A2, leaving LOGO macro accuracy at `.9479`.
On v16 it sends all A1 rows to A2, splits A2, leaks `13/11` P rows, and gives macro `.25` under
both oracle folds; the folds disagree.  Runtime simultaneous-sign invariance holds within
`3.40e-6`, so this is not a direction-gauge implementation failure.  Per the prospective kill
rule, contextual centroid feature engineering is closed: neither response magnitude nor fixed
projective response shape identifies the task branch.

The route should now change circuit target.  The v15 rank-two state program already has selective
execution, direct residual transport, exact decoding, removal, sufficiency, and gain-calibrated
payload reuse.  Its highest-information unsettled property is joint task composition/reuse, not a
third router representation.  A new test should combine the independently defined temporal and
is-was commands at their causal state interfaces and preregister additive versus interacting
predictions before opening joint outcomes.

The joint-composition compatibility audit separates a useful common interface from two missing
prerequisites.  The temporal Q8 realization and the v15 is-was direct route both terminate in the
same 1152-dimensional final residual and use the same final RMSNorm, tied unembedding, and soft cap.
However, neither immutable result stores an interoperable physical command basis, and repository
search finds no candidate population in which temporal and is-was commands are changed in the same
row.  Existing cross-task matrices concatenate separate task rows; they are not simultaneous
composition evidence.  The next legal sequence is therefore one zero-update common-gauge basis
capture and a native-only dual-command capability builder before a 2x2 causal factorial.

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
- Exact selected-projector weight translation:
  `temporal_iswas_v15_selected_head_projector_weight_interfaces_v1_result.json` (immutable first
  artifact, absolute-tolerance invalid) and
  `temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit_result.json`
  (valid zero-model scale-aware repair; diagnostic, not causal identification).
- Attention-15 causal-path scope:
  `temporal_iswas_selected_projector_attn15_reader_path_audit_v1_result.json` (the complete
  attention-15 clamp bypasses computed L15H5 head output, so the weight-ranked interface is not
  causally tested by current DAS outcomes).
- Native attention-15 dependency:
  `temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json`
  (valid live-route candidate; all six predictions pass), followed by
  `temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_admission.json`
  (zero-model admission of the full module/nine-head mediation atlas).
- Layer-15 head/module mediation:
  `temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_result.json` (valid bypass null;
  L15H5 is the top but small singleton, while the complete module fails the 10% mediation gate).
- Residual-suffix complete-write mediation:
  `temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1_result.json` (valid direct-residual
  carry null; the joint layer-12--17 write bank mediates only `.141-.207` reset and `.127-.210`
  rescue, with no stable singleton module).
- Ordered residual-state mediation:
  `temporal_iswas_v15_residual_state_boundary_mediation_atlas_v1_result.json` (valid complete-state
  interface; exact full mediation at every boundary, with the known parity-0 P collision and no
  later consumer transition localized).
- Entry-12 shared target/control bases:
  `temporal_iswas_v15_entry12_shared_target_control_response_basis_v1_result.json` (valid rank-two
  construction union; exact P complementation removes controls but destroys target, and rank-one
  pooled DIM is not construction-general).
- Entry-12 native-state finite router:
  `temporal_iswas_v15_entry12_crossfit_finite_router_v1_result.json` (valid router-identification
  null; the gold finite mixture composes, but native base state cannot distinguish paired A1/P rows
  whose base texts are identical).
- Entry-12 source-delta centroid router:
  `temporal_iswas_v15_entry12_crossfit_source_delta_router_v1_result.json` (valid identification
  null; predicts off for every held row while the gold routed union remains target-effective and
  exactly control-selective).
- Entry-12 regularized source router:
  `temporal_iswas_v15_entry12_crossfit_regularized_source_router_v1_result.json` (nested CV is
  perfect inside one signed tense direction but the opposite direction collapses entirely to off;
  linear raw-delta routing is not direction-gauge invariant).
- Entry-12 Gram router:
  `temporal_iswas_v15_entry12_crossfit_gram_router_v1_result.json` (both targets pass and one parity
  routes perfectly, but three off rows leak to A1 in the other parity; the coarse response-signature
  route is closed by its preregistered strict control gate).
- Entry-12 unordered token-pair router:
  `temporal_iswas_v15_entry12_unordered_token_pair_router_v1_result.json` (selective v15 executable
  screen: perfect held routing, `.795-.881` target projection, zero P/C effect, and exact gold
  composition), followed by
  `temporal_iswas_v15_token_router_v16_signature_coverage_audit_v1_result.json` (valid zero-model
  OOD null: none of the 32 native-capable v16 target rows shares either learned v15 signature, so
  the immutable default-off rule predicts exactly zero transfer; v16 C is explicitly outside the
  exact aligned-signature program because its base/donor token lengths differ).
- Entry-12 normalization-aware weight atlas:
  `temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1_result.json` (preserved
  invalid v1 because the registered six-forward price omitted four manual reference executions),
  followed by
  `temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit_result.json`
  (valid zero-model price correction to ten forwards).  RMS-Jacobian responses track exact finite
  responses for every interface; exact rankings are stable across folds and nominate shared
  L12H0:v, L12H4:q/k, L13H2:q, and L15H5 interfaces.  Raw and exact top tens overlap `.9-1.0`,
  so normalization does not change the coarse ranking here.  All four known oracle source heads
  rank above the 97th upstream-writer percentile.  These are weight nominations, not causal readers.
- Entry-12 layer-12/13 attention factor mediation:
  `temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1_result.json`
  (valid weight-reader bypass null).  Complete attention12/13 factor banks mediate only
  `-.0014-.0118` of the selective rank-two effect; no complete head or nominated factor passes.
  Singleton heads reproduce each module reset vector with cosine above `.999997` and relative L2
  below `.0042`, so hidden head interaction is not the explanation.  Static weight response is not
  causal reader identification for these modules.
- Entry-12 direct residual/final-head route:
  `temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json` (valid explicit
  route).  With every layer-12--17 attention/MLP write frozen native-off, the selective program
  retains `.803-.871` signed projection with direction `1.0`; suffix writes contribute `.129-.201`.
  The final residual delta equals the entry12 delta times the six recurrent `lambdas[0]` product
  `1.51363146` within `1.42e-6` relative L2, after which final RMSNorm, tied unembedding, and soft cap
  reproduce logits exactly.  This identifies a dominant v15 computational path, not its semantic
  OOD router.
- Final-state rank-two removal/sufficiency/swap:
  `temporal_iswas_v15_final_rank2_removal_construction_swap_v1_result.json` (valid selective
  manipulation with an operational construction-gain split).  Removing the transported span loses
  `1.106-1.212` of the live target response and projection-only states recover `1.133-1.198`; on the
  frozen direct path both are `1.0` to numerical precision.  Unscaled A1/A2 payload swaps preserve
  direction and cosine (`>=.995`) but fail magnitude: A2 payloads overshoot A1 (`1.61-1.98`) while
  A1 payloads undershoot A2 (`.557-.758`).  The shared direction therefore needs a construction
  gain before payload reuse is established.
- Cross-fit construction-gain swap:
  `temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1_result.json` (valid partial null).
  Opposite-parity state-norm gains are reciprocal and cross-fold stable within `19.6%`.  They repair
  the frozen direct path to `.807-1.234` signed projection with cosine `>=.995`, but the live suffix
  still overshoots two cells to `1.342/1.389`, and one cell worsens relative to the unscaled swap.
  Thus the dominant direct payload is reusable with a scalar construction gain, while the minor
  live suffix correction is construction/parity dependent; the registered whole-program claim fails.
- Frozen tied-embedding semantic router:
  `temporal_iswas_v15_tied_embedding_semantic_router_v16_v1_result.json` (valid in-distribution and
  OOD null).  Both folds recover all v15 targets but misroute every held P row to A2, giving macro
  accuracy `.8333`; both then default every v16 A1/A2/P row to off, giving macro accuracy `1/3`.
  Simple unordered tied-embedding midpoint/squared-difference geometry is neither selective on v15
  nor target-covering on v16.
- Multi-direction contextual Gram router:
  `temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1_result.json` (valid strict null).
  Pooling both v15 directions yields LOGO macro `.9479` but retains three control false positives.
  V16 macro falls to `.1458/.125` under the two oracle folds, A1/A2 are almost wholly exchanged,
  P leaks `9/11` rows, and fold predictions disagree.  Six coarse contextual response moments do
  not identify a stable semantic branch.
- Projective contextual response-shape router:
  `temporal_iswas_v15_response_shape_router_v16_v1_result.json` (valid strict null).  V15 LOGO
  classifies all targets but leaks five controls to A2 (macro `.9479`).  Both v16 folds have macro
  `.25`; all A1 rows route as A2, A2 is split, `13/11` P rows leak, and folds disagree.  The fixed
  response-shape object is sign invariant to `3.40e-6`, so contextual centroid feature engineering
  closes rather than expanding.
- Joint-command composition compatibility:
  `temporal_iswas_joint_command_composition_compatibility_audit_v1_result.json` (valid zero-forward
  scope result).  Both programs share the final-residual decoder boundary, but stored results lack
  interoperable physical bases and no genuinely simultaneous dual-command builder exists.  Earlier
  cross-task Hankel and task-mode rows remain separate-command evidence, not joint composition.
- Common-final-gauge basis capture:
  `temporal_iswas_common_final_gauge_basis_capture_v1_result.json` (valid task-typed direct-sum
  result).  The two cross-fitted is-was rank-two spans are stable, with principal cosines
  `.9050/.8087`, but are not contained in temporal Q8: their Q8 cosines are only
  `.6752/.2465` and `.6520/.1941`.  The physical union has rank 12.  All 1152x8/1152x2 bases are
  now stored byte-exactly in one final-residual gauge, so weight translation can use the actual
  tensors; joint work must preserve task type rather than calling this one shared subspace.
- First same-sequence dual-command capability:
  `temporal_iswas_dual_command_native_capability_v1_result.json` (valid native capability null).
  Is-was is perfect in all 16 cells, and future temporal cells are perfect, but past temporal cells
  fail symmetrically across FIT/HOLDOUT: the forecast template is `0/8` and schedule template is
  `4/8`.  No row is licensed and no causal outcome opens.  The bank is closed without filtering.
  A successor may preserve an already capability-qualified temporal prefix exactly—causality makes
  its earlier-token margin invariant to any appended second clause—then test only whether the later
  is-was command survives the composition.
- Prefix-preserved same-sequence capability:
  `temporal_iswas_dual_command_prefix_preserved_native_capability_v1_result.json` (valid complete
  license).  All 32 phase/template/cell/role gates pass, with minimum cell accuracy `.875`, maximum
  `1.0`, and mean `.9961`; all 32 rows are licensed without filtering.  Preserving the exact
  standalone temporal prefix eliminates the first bank's past-tense failure while the appended
  is-was command remains capable.  This opens the preregistered full-head/module causal factorial.
- Common-gauge exact-weight interface atlas:
  `temporal_iswas_common_gauge_weight_interface_atlas_v1_result.json` (full valid screen).  Despite
  the rank-12 task-typed state, weight interfaces substantially overlap: is-was fold top-20 Jaccards
  are `.8182` for readers and `.7391` for writers.  Six interfaces exceed fourfold isotropic
  enrichment for temporal Q8 and both is-was folds: writers L9H1, L11H3, and L15H5, plus L11H3:v
  and L15H5:q/q2 readers.  L11H3 is strongest for temporal writing (`24.38x`) and value reading
  (`13.80x`); L15H5 is strongest for is-was writing (`12.51/12.54x`) and q2 reading
  (`7.79/7.46x`).  This is exact physical weight compatibility and a sharply reduced intervention
  set, not causal identification by itself.
- Dual-command shared head/module factorial:
  `temporal_iswas_dual_command_shared_head_module_factorial_v1_result.json` (valid registered
  causal-route null with a strong distributed fragment).  No single complete parent module reaches
  the frozen `.50` recovery bar for both phases, so B/C fail and the terminal remains null.  Yet
  every singleton and module has positive, high-cosine, perfectly directional effects.  The fixed
  L9H1+L11H3+L15H5 union recovers `.706/.752` of the temporal toggle and `.523/.491` of the is-was
  toggle on FIT/HOLDOUT, with cosines `.991-.998`, direction `1.0`, and other-role norm ratios at
  most `.00288`; later is-was patches change the earlier temporal margin exactly zero.  Union-minus-
  singleton-sum error is only `.024-.057`, so the preregistered distributivity clause passes.  This
  does not retro-pass the failed singleton/module identification claim; it licenses the distributed
  three-head combination for the next joint composition test.
- Distributed three-head joint composition:
  `temporal_iswas_three_head_union_joint_composition_v1_result.json` (valid full license).  The
  L9H1+L11H3+L15H5 union exactly reproduces its parent metrics and, when temporal and is-was swaps
  are installed simultaneously at their distinct answer queries, preserves temporal recovery
  `.706/.752` and is-was recovery `.523/.491`.  Every phase/template interaction ratio is at most
  `.01130` and additive cosine is at least `.999936`; temporal output has exactly zero interaction,
  later-to-earlier causal effect is exactly zero, and single-arm cross-role collateral remains
  below `.00288`.  This identifies a shared physical three-head interface carrying task-typed
  command states compositionally.  It does not identify any singleton module as sufficient and is
  still a partial-effect circuit rather than a full behavioral replacement.
- Greedy shared-writer augmentation:
  `temporal_iswas_greedy_shared_writer_augmentation_v1_result.json` (valid full license).  FIT
  selects the smallest qualifying addition, L9H4, yielding the four-head union
  L9H1+L9H4+L11H3+L15H5.  Temporal recovery rises from `.706/.752` to `.833/.848` and is-was from
  `.523/.491` to `.696/.669` on FIT/HOLDOUT; summed gains are `.299/.275`, cosines are
  `.995-.999`, direction is `1.0`, and collateral is at most `.00369`.  Adding L8H1 too would reach
  `.859/.864` temporal and `.855/.840` is-was, but is not selected because the frozen rule prefers
  the smallest arm already meeting both quality bars.  The selected H4 now requires simultaneous
  composition confirmation.
- Four-head joint composition:
  `temporal_iswas_four_head_union_joint_composition_v1_result.json` (valid full license).  The
  selected L9H1+L9H4+L11H3+L15H5 program exactly replays its single-command result and preserves
  temporal recovery `.833/.848` and is-was `.696/.669` under simultaneous installation.  Temporal
  interaction is exactly zero.  Is-was interaction ranges only `.00297-.01118` across held-out and
  template panels, with additive cosine at least `.999938`; later-to-earlier causal effect remains
  exactly zero.  H4 therefore supersedes H3 as the best currently licensed joint head program,
  while remaining a partial-effect rather than complete behavioral replacement.
- H4 exact reader-factor split:
  `temporal_iswas_h4_reader_factor_factorial_v1_result.json` (valid partial split).  L11H3:v over the
  causal prefix explains essentially all of its complete-head effect in both tasks and phases:
  parent-relative recovery is `1.015/1.026` temporal and `.966/.973` is-was, cosine `.993-.997`,
  and direction `1.0`.  By contrast, the weight-nominated L15H5 q/q2 story is causally false in this
  intervention: q is anti-aligned and negative (`-.118` to `-.225`), q2 is small and positive
  (`.055-.226`), and q+q2 nearly cancels (`-.053` to `.012`).  Their pooled is-was interaction ratio
  is `.771/.618`, so the multiplicative pattern factors cannot be called an additive reader split.
  The three-factor union passes only because L11H3:v dominates its two-head parent.  Preserve the
  L11 value-reader identification, close L15 q/q2 as a causal explanation, and do not convert exact
  weight enrichment into causal identity without intervention.
- Fresh dual-command OOD capability:
  `temporal_iswas_dual_command_ood_native_capability_v1_result.json` (valid complete native license).
  All 32 cells pass on temporal-v12 registry/field-report plus is-was-v16 right-now/these-days
  constructions and sixteen fresh lexical groups.  Thirty-one cells are `8/8`; FIT
  field-these-days past-temporal is `7/8`.  All rows are retained.  This opens a prospective H4 OOD
  intervention, but native capability alone is not circuit generalization evidence.
- H4 fresh-template/lexicon joint composition:
  `temporal_iswas_h4_ood_joint_composition_v1_result.json` (valid full license).  With no head or
  threshold reselection, H4 recovers `.835/.833` of temporal and `.730/.739` of is-was command
  effects on OOD FIT/HOLDOUT.  Cosines are `.997-.999`, direction is `1.0`, and cross-role
  collateral is at most `.00342`.  Temporal interaction remains exactly zero; is-was interaction
  is only `.00365-.00646` across fresh templates, with additive cosine at least `.999979`.
  Simultaneous recovery is unchanged and later-to-earlier effect is zero.  This establishes stable
  identification of the H4 causal interface across new constructions and lexicons; selective
  removal and standalone extraction remain untested.
- H4 selective midpoint removal:
  `temporal_iswas_h4_selective_midpoint_removal_v1_result.json` (valid full license).  Clamping each
  H4 command slice to the pair-symmetric native midpoint recovers the prospectively expected
  half-toggle at `.828/.847` temporal and `.683/.652` is-was on original FIT/HOLDOUT, and
  `.835/.831` temporal and `.714/.722` is-was on OOD.  Cosines are at least `.991`, direction is
  `1.0`, every native-correct endpoint moves toward a smaller correctness-aligned margin, and
  collateral is at most `.00352`.  Pair midpoint invariance and later-to-earlier causal zero are
  exact.  Simultaneous removals preserve both effects; temporal interaction is zero and is-was
  interaction is at most `.01043`.  H4 is therefore selectively manipulable under a frozen
  contrast-removal edit, in addition to being sufficient for high-recovery swaps.  Standalone
  extraction and literal simplicity remain open.
- L11H3 value source-region localization and replay audit:
  `temporal_iswas_l11h3_value_source_region_localization_v1_result.json` remains immutably
  `invalid` because its A gate compared parent-relative full-effect cosine (exactly one) with the
  reader-factor receipt's command-gold cosine.  The separately registered three-forward audit
  `temporal_iswas_l11h3_value_source_region_localization_v1_replay_audit_result.json` reproduces
  the reported erroneous maximum exactly (`.028743869178895265`) while every like-for-like
  command-gold metric and both already-stored command-gold fields replay with zero error.  The
  audit therefore licenses the frozen B--E evidence without changing the original terminal or A:
  semantic regions compose with low error, pre-cue is inert, both selected regions validate on
  original HOLDOUT and both OOD phases, and the source routes are task typed.  Temporal selects
  the unchanged bridge (`.996` original HOLDOUT; `1.026/1.029` OOD FIT/HOLDOUT), whereas is-was
  selects postcue (`.843`; `.849/.852`), all with direction `1.0` and high cosine.  L11H3:value is
  now localized to stable but different semantic source regions; its query/key routing partner
  remains untested.
- L11H3 task-typed source QK factorial:
  `temporal_iswas_l11h3_task_typed_source_qk_factorial_v2_result.json` (valid stable routing
  invariant).  The amended run replays every selected-value parent metric with zero error and its
  16-corner Möbius/Shapley accounting is exact.  Is-was donor q/k/q2/k2 changes the postcue-value
  parent by only `.081` on original FIT and `.056-.074` on validation, so no command-varying QK
  factor is selected.  Temporal barely clears the `.10` FIT relevance gate (`.101`) and selects
  k+q+q2; this subset predicts the full routing delta on original HOLDOUT and OOD
  (`.785-.952` recovery, cosine `.983-.998`), but the full delta itself is only `.081-.123` of the
  value-parent norm.  All registered arm collateral is <=`.00474` and later-to-earlier effect is
  zero.  Recipient-native attention routing therefore transports most of the task-specific value
  effect; donor QK changes are a small temporal modulation, not a shared command subspace.  The
  next object is the exact native-routing bilinear source term, not another hybrid-factor sweep.
- L11H3 exact native-routing source-term extraction:
  `temporal_iswas_l11h3_native_routing_source_term_extraction_v1_result.json` (valid full
  extraction license).  For every destination, the explicit tensor
  `(1-lambda_11) sum_{j in R} ((q_i·k_j)/128)((q2_i·k2_j)/128) delta_v_j` reproduces the
  independently observed c_v-patch L11H3 preprojection delta with maximum absolute error
  `9.54e-6` and relative L2 `1.20e-7`.  Adding that tensor directly to the native L11H3
  preprojection slice, with no c_v hook, reproduces selected answer logits within `8.59e-6` and
  every phase/template causal effect at recovery `.9999988-1.0000004`, cosine at least
  `.99999999999`, and residual at most `4.52e-6`.  Command-gold replay error and later-to-earlier
  causal effect are zero; collateral is at most `.00474`.  Four 128x128 query-tensor banks with
  row, query, and source metadata are stored and checksummed in the result.  This is an exact,
  directly executable, original/OOD-stable interface—not a fitted subspace—and it opens physical
  upstream writer translation through L11H3 W_v.
- Source-tensor upstream exact-weight atlas:
  `temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_v1_result.json` (valid shared-candidate
  atlas).  All 111 earlier writers were composed through the physical L11H3 W_v slice and scored
  against the four trace-one extracted-tensor covariances.  Original/OOD top-ten Jaccard is `1.0`
  for both tasks.  Temporal top five is L7H7, L9H4, L9H1, L6H7, L5H1; is-was top five is L7H7,
  L9H4, L6H7, L9H1, L3H4.  Four writers are prospectively shared: L6H7, L7H7, L9H1, and L9H4.
  Their enrichments range from `6.11x` to `17.25x` across all four task/population panels; L7H7 is
  strongest overall.  This makes the user-proposed tensor-to-weight translation operational and
  freezes a six-head causal source-position union.  It is exact weight compatibility, not causal
  writing, until that head patch test lands.
- Source-tensor upstream causal head factorial:
  `temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1_result.json` (valid shared-writer
  result).  All six weight-nominated heads were patched singly and as each task's frozen top-five
  union at only the locked source positions, against a live L11H3:value reference.  L9H1 and
  L9H4 independently pass the complete cross-task singleton gate.  The temporal top-five union
  recovers `.554/.562` on original FIT/HOLDOUT and `.575/.585` on OOD; the is-was union is
  directionally faithful but overcomplete, recovering `2.305/2.501` original and `2.914/2.901`
  OOD.  Pooled union cosines are `.970-.996` and direction agreement is `1.0`.  Across every
  phase/template, singleton sums predict the simultaneous union with maximum relative L2
  `.17425` and minimum cosine `.98560`, so the registered distributivity branch passes.
  Reference replay and later-to-earlier causal effect are exact zero.  The weight atlas is now
  causally validated, but the five-head is-was set should be pruned rather than treated as the
  final circuit.  The licensed successor is weight-ordered greedy prefix selection on original
  FIT followed by unchanged original-HOLDOUT/OOD validation.

### Aligned v23 four-head is–was program and weight anatomy

The weight-ordered and complete-module continuation has now moved the is–was branch beyond the
six-head source-factorial screen.  The important correction is that two different interfaces had
been mixed: a shared L11H3 value-source circuit and a broader behavior-carrying four-head program.
The latter is now identified on a fully token- and endpoint-aligned fresh bank.

- The v21 complete upstream atlas and v22 reader-contracted effect game nominated
  `L8H1`, `L9H1`, `L9H4`, and `L11H3`.  V22's target and P effects were informative, but its
  canonical C comparison was invalid because C endpoints differed.  The history-disjoint v23
  corpus repaired that instrument rather than reinterpreting the invalid control.
- `temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json` is the
  valid selective confirmation.  The union recovers `.72588` of behavior and `.67007` of the
  frozen M11-reader-contracted response, with cosines `.99524/.97676`, direction `1.0`, P leakage
  `.10033/.08344`, aligned-C leakage `.00396/.00640`, and exact self/hidden closure.  Every head
  passes its registered singleton screen and every Shapley allocation is positive and stable
  across reporter halves.  This is a distributed four-head writer program, not a selected rank.
- Exact transport splits the native heads rather than treating them as copies.  In
  `temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json`, the state-dependent RMS
  tangent predicts adjacent L11H3 transport with cosine `.99998`, signed projection `.99237`, and
  residual `.00945`; raw `W_O -> M11` composition is invalid.  The L9 and L8 localization results
  show L9H4 is nearly transport-ready after block 9, L9H1 needs block-10 rotation plus block-11
  attention gain, and L8H1 is progressively oriented by blocks 9/10 before the same late gain.
- `temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json` establishes reverse
  necessity: four-head removal recovers `.72151` of the native target effect with cosine `.99401`,
  direction `1.0`, P/C leakage `.11861/.00416`, and `.72567/.71771` half projections.  Its stored
  terminal label is superseded by the adjacent interpretation audit.  Restoring the entire induced
  M11 factor difference rescues only `.15114` of the head-removal behavior.  The leading occupied
  weight mode explains `.87376` of that small M11-local rescue, but direct mode removal reaches only
  `.18197` of the full head effect.  Therefore M11 is a minor local branch; neither a dominant nor a
  multidimensional M11 mediator is licensed.
- The exact restricted weight tensor in
  `temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json` separates
  checkpoint capability from realized use.  Literal M11 factors are broadly shared across heads
  (median nonnegative factor-score cosine `.98538`) but diffuse (top-256 squared mass `.28090`) and
  do not form one signed head map (leading head-mode energy `.3012`).  Activation occupancy is much
  narrower: `.97137` of the realized M11 response lies in one frozen reader mode, with reporter-half
  profile cosine `.99998`.
- Contracting the exact weight tensor with that occupied reader mode does not license wholesale
  cross-head merging.  The literal-coordinate result has leading head-mode energy `.29723` and map
  cosine `.04860`.  Its immediate gauge audit is authoritative for interpretation: direct comparison
  of different heads' private 128-dimensional coordinates is not rotation invariant.  Gauge-invariant
  context-Gram similarity is high (`.92455`) and the leading context direction overlaps about `.96`,
  while full-map Procrustes similarity is only `.68596`, rank-8 context overlap `.39605`, and rank-32
  overlap `.26262`.  The prospective grouping hypothesis is consequently a shared low-dimensional
  context-read core with head-private adapters and tails.  No rank was selected from this opened
  ladder, and no complete common weight map is claimed.

The exact block-11 residual-versus-M11 factorial is currently hash-bound in the managed queue.  It
adds back the post-attention residual delta, the M11 contribution, or their exact sum after four-head
removal.  Its successor complete-module reader atlas was registered outcome-blind and already has a
tested reciprocal transfer/reset executor.  These pending outcomes decide whether the dominant branch
travels by residual skip to the final literal RMSNorm/unembedding or is rewritten by a downstream
module.  Queue presence is not evidence for either outcome.

## Remaining gates

1. Score the queued exact block-11 residual/M11 factorial.  Joint restoration must close exactly;
   the separately rescuing branch, not M11 occupancy alone, determines the next reader object.
2. Run the frozen reciprocal downstream module atlas.  Any attention module must pass both transfer
   sufficiency and reset necessity on unchanged HOLDOUT before all nine physical heads are split.
   A clean singleton null opens the already demonstrated recurrent residual-product law and exact
   final RMSNorm/unembedding route on aligned v23.
3. Freeze the partial shared-context-core hypothesis on a genuinely fresh authority before choosing
   its dimension.  Require causal interchange with private head adapters and P/C selectivity; the
   opened weight spectrum alone cannot select or identify the group.
4. Test the resulting program on fresh/OOD constructions and compose it with the existing temporal
   command circuit.  Preserve later-to-earlier causal zero and both task endpoints under simultaneous
   removal, swap, and edit.
5. Price a standalone extracted executor only after residual/reader fidelity, OOD prediction, and
   composition are verified.  Keep every invalid endpoint-control and gauge-dependent interpretation
   in the record rather than selecting a favorable instrument after outcomes.
