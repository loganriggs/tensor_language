# Block-3 native-gate subset family F — consequence-fit implementation amendment

Frozen after activation-fitted family A failed validation V1 and before any family-F
fit row, teacher logit, candidate write, gradient, score, support, or fitted affine
parameter is loaded or computed.  The registered V1 action is
`stop_activation_family_and_preregister_finite_suffix_family`; this document implements
that branch.  It does not reopen family A, alter K, or authorize final-role access.

## Question

Family A minimized symmetric local error in four stacked polynomial terms.  Validation
showed that individually useful typed replacements fail when all four compose, and that
the mirror sign of the same local error is much less harmful.  Family F asks whether
the same finite native-product grammar becomes useful when gate importance is learned
through the actual frozen nonlinear suffix:

$$
\text{MLP3 write}\longrightarrow\text{Blocks 4--17}\longrightarrow
\text{next-token distribution}.
$$

Only the fit role may train F.  Validation remains an already-exposed selection role;
the fixed final role stays sealed unless a prospective F-validation transaction passes
the original consequence gates.

## Frozen data and tensor coordinates

- fit rows: canonical collision-separated `n480_skip80`, 480 rows from 209 source
  documents;
- tokens entering the model: positions 0--255;
- scored teacher/student logits: positions 64--255;
- fixed logical batches: consecutive groups of 8 rows;
- backward microbatch: 2 rows, four microbatches per logical batch;
- model/program dtype: float32 on CUDA; continuous score/Adam state and capped-simplex
  projection dtype: float64;
- native product coordinates: the positive norm-balanced gauge;
- candidate universe: the already sealed 1,024 contribution-energy prefilter;
- budgets: nested K=256 and K=512.

Every row is weighted by the reciprocal of the number of cached rows from its source
document.  The epoch objective is therefore a mean over 209 source documents, not a
mean over 480 possibly repeated rows.  No fit/validation/final document overlaps.

## Teacher consequence

For each logical batch, run the exact native prefix through MLP3 and an autonomous
native suffix through Blocks 4--17.  Detach the soft-capped native logits.  A student
starts from the same post-attention-3 residual, initial residual, and attention
first-value state, substitutes its MLP3 write at **all** positions 0--255, and reruns
Blocks 4--17 with no teacher-state reuse.

For row $r$, let $p_r$ be the native next-token distribution and $q_r$ the student
distribution on positions 64--255.  The consequence loss is

$$
\mathcal L = \frac1{209}\sum_d\frac1{n_d}
\sum_{r\in d}\frac1{192}\sum_t
\operatorname{KL}(p_{rt}\Vert q_{rt}).
$$

Per-position logit translations are automatically quotiented by softmax.  Training
targets are teacher distributions, never ground-truth validation/final tokens.
Teacher and student suffixes return raw `lm_head` logits.  The model's
`30*tanh(raw/30)` softcap is applied exactly once, after selecting positions 64--255;
already-softcapped facade/model outputs are forbidden inputs to that function.

## Stage F1: decoder-scale-fixed continuous gate scores

Let $P$ be the fixed 1,024-gate prefilter and $\phi_P(z)$ its balanced native product
features.  The selection student uses the **fixed native Down columns** and continuous
scores $s\in[0,1]^{1024}$:

$$
w_s(z)=b+D_P\bigl(s\odot\phi_P(z)\bigr).
$$

Fixing $D_P$ removes the explicit score/decoder rescaling degeneracy: a score cannot be
absorbed into a freely changing decoder column.  Correlated/duplicate gates and
suffix-null directions can still make individual scores non-identifiable, so scores
rank supports and receive no atom-level causal interpretation.  Scores are constrained
to the capped simplex

$$
0\le s_i\le1,\qquad\sum_i s_i=512.
$$

Initialize all float64 scores to 0.5.  Optimize with projected Adam for exactly 8 epochs = 480
logical steps: learning rate 0.02, betas (0.9,0.999), epsilon `1e-8`, zero weight decay,
global gradient-norm clipping at 1.0.  After each Adam step, project in Euclidean norm
onto the capped simplex by 64 deterministic bisection iterations and require
`abs(sum(s)-512)<=1e-10` in float64 plus the KKT boundary conditions.  No post-cast
residual redistribution is allowed.  No early stopping or validation-dependent
schedule is allowed.

For one logical eight-row batch, zero gradients once, backpropagate the four additive
two-row microbatch contributions, clip once, call `Adam.step()` once, and project once.
Calling Adam or projecting per microbatch changes the registered 480-step trajectory
and is forbidden.  Runtime must require exactly four losses, a PyTorch Adam optimizer,
and exact identity equality between the parameters passed for clipping and all
parameters owned by the optimizer.  The float64 sum, box, and clamp-form KKT
conditions are checked after every projection step, not only in a unit test.

Train three score vectors from the same initialization:

1. `teacher`: the aligned detached native teacher distributions;
2. `teacher_row_reversal`: teacher logits reversed across the eight rows of every
   fixed logical batch, without changing student prefixes or document weights.  This
   preserves the original frozen reversal but is only a weak correlated row-label
   support-selector null: 132/480 target/donor pairs share a source document and seven
   logical batches contain one document.  It is nonpromotive and cannot support a
   negative claim about document-independent label information;
3. `teacher_document_derangement`: order the 209 documents by first row; document `d`
   receives teacher labels from `(d+104) mod 209`, and target occurrence `j` receives
   donor occurrence `j mod n_donor`.  It has zero target/donor document matches and is
   also a nonpromotive support-selector diagnostic.

Supports are selected by descending final score, with global native gate index as the
tie breaker.  The first 256 and first 512 supports are nested.

## Stage F2: registered joint decoder refit

For each real-teacher selected support, refit one decoder analytically from the sealed
four-term fit Gram/cross statistics, exactly as family A:

$$
\widehat D_S^T=(G_S+10^{-6}\,\operatorname{tr}(G_S)/K\,I)^{-1}C_S.
$$

This follows the original family-F wording: suffix consequences choose the finite gate
support; the common four-term decoder is then refit.  The refit uses fit statistics
only.  For both budgets save:

1. the promotive real-teacher F support with the real local cross moment;
2. seed-2026082907 matched-random support with the real local cross moment;
3. the **registered same-support label-permutation control**: the real-teacher F support
   with the sealed reversed-label `prefilter_permuted_cross` decoder;
4. reversed-row and document-deranged selector supports with real local decoders,
   explicitly nonpromotive.

The same-support permuted-cross program is the matched label control.  The two
different-support selector nulls must never substitute for it.  Report teacher KL of
the continuous fixed-Down F1 score program, the binary native-Down support before
refit, the post-refit deployed program, and both changes; F1 KL is not mislabeled as
the deployed program's KL.  All deploy through one direct K-product bank, not four
typed banks.

## Stage F3: zero-marginal-deployment-cost affine diagnostic

The user-proposed cheap correction is tested explicitly rather than hidden inside a
decoder.  At K=512 only, separately calibrate the real-teacher F program, sealed family
A program, matched-random program, and registered same-support permuted-cross program:

$$
w_{a,c}(z)=b+c+a\,[w(z)-b]
$$

against the same fit teacher consequence.  Initialize $a=1,c=0$.  Optimize only the
scalar and 1,152-vector for exactly 4 epochs = 240 logical steps per arm with Adam, learning
rate 0.005 and otherwise the same optimizer/microbatch/gradient-clipping rules.  Fold
$a$ into the decoder and $c$ into the already stored bias vector at publication.  Thus
neither changes deployed bytes, products, multiplies, or additions: calibration changes
the values of two already required arrays but introduces no new array or operation.

Calibration nevertheless adds 1,153 fitted degrees of freedom and 960 logical optimizer
steps across four arms: 4,612 fitted coordinates in total across the four separate
diagnostic fits.  It is not zero MDL/search/fit cost.  Calibrated arms are
diagnostic and **nonpromotive**: only the uncalibrated F program with native bias and
the registered joint local decoder may satisfy family F or open final.  In particular,
calibrated family A cannot reopen family A after its observed validation failure.

The affine scalar, correction vector, gradients, and Adam state are float64 during
fitting.  Each fitted scalar/vector is folded into a fresh float32 copy of the already
stored decoder/bias for replay and publication; no float64 parameter is deployed.

## Frozen fit reporting and failure bars

Report per epoch and arm:

- document-balanced teacher KL;
- unweighted row KL as a diagnostic;
- score min/max, capped-simplex sum, fraction within `1e-6` of 0 or 1;
- gradient norm before clipping;
- overlap/Jaccard with family A and matched random at K=256/512;
- local stacked NRMSE after decoder refit;
- affine scalar and correction-vector RMS/norm;
- exact literal bytes, products/token, and multiplies/token;
- direct versus polarized replay;
- measured prefix, teacher-suffix, student-suffix, backward, and outer-model call counts;
- the exact donor-row reuse multiplicity vector for the many-to-one document-
  derangement diagnostic;
- wall time and maximum allocated CUDA memory.

Known-answer checks:

1. native autonomous suffix must replay the ordinary model before optimization;
2. replacing teacher logits by themselves gives KL below `2e-7` per token;
3. the score vector remains finite and on the capped simplex every step;
4. published programs replay their stored supports, decoder, bias, and direct K-product
   output exactly within the V1 relative replay tolerance;
5. model tensors and all sealed inputs are byte-identical before/after.

If aligned score loss is nonfinite, no program is published.  A finite run is retained
even if loss does not improve; lack of convergence is a scientific failure, not a
reason to silently change epochs or learning rate.  Family F receives no credit from
fit loss alone.

After all parameters are frozen, make exactly one no-gradient fit-reporting sweep over
the 480 rows in 60 batches of eight.  A batch computes one target prefix and one real
native teacher suffix, then 18 student suffixes: the final continuous real-teacher F1
score program; at each K, binary-native-Down and post-refit programs for real F plus
post-refit random, same-support permuted-cross, reversed-selector, and document-
selector programs (12 arms total across two K); uncalibrated family A at K512; and the
four folded affine K512 diagnostics.  Thus reporting adds exactly 60 prefix calls, 60
teacher suffixes, 1,080 student suffixes, and no native student-MLP3 call.  Prefix and
teacher tensors may be reused only within that reporting batch and are deleted before
the next batch.  No row, prefix, or teacher logit is retained in a published artifact.

## Frozen family-F validation and advancement rule

This fit does not open validation.  A separate audited transaction evaluates only the
**uncalibrated real-teacher F** programs at K=256 then K=512 as promotive candidates.
Bias-only, native, mirror, uncalibrated matched random, and the registered uncalibrated
same-support permuted-cross program are the matched controls.  Calibrated arms,
family-A arms, and both different-support selector nulls are diagnostic and cannot
advance.

Because two K values are tested sequentially on the already-exposed validation role,
replace every original one-sided bootstrap q95 promotion bound with q97.5 while keeping
the original numerical threshold.  A candidate must pass all original local, KL, CE,
singleton, and mirror gates; beat both matched controls on point KL ratio; and have the
q97.5 paired source-document-bootstrap bound for
`candidate KL ratio - control KL ratio` below zero for each control.  Select the
smallest K satisfying every condition.  No averaging, tradeoff, or post-outcome choice
between calibrated and uncalibrated arms is allowed.

If neither K passes, family F stops and final stays sealed.  If one passes, complete the
full 16-replacement/15-omission cube on validation for that uncalibrated program; it
must pass the original interaction gates with the q97.5 substitution before the
untouched fixed final role opens.  Final repeats the full cube and original final bars
without calibration.  All validation claims are labeled family-sequential; only final
is terminal replication.

## Lifecycle and resource ceiling

The family-F authority is create-only and published before fit rows or teacher outcomes
load.  It binds this amendment, all executable sources/tests, the A fit artifacts, V1
validation result/receipt and registered branch, row receipt, checkpoint, optimizer,
document map, and exact source commit pushed to `origin/main`.  Programs/results are
published before a receipt-last terminal artifact; any drift writes failure and no
receipt.

Resource ceiling: 45 wall-clock minutes and 30 GiB allocated CUDA memory.  Exceeding
either publishes a failure without changing the algorithm.  No validation or final row
loader may be imported or invoked by the fit transaction.

The literal optimization schedule is 3 score arms x 480 = 1,440 logical score steps and
5,760 two-row backwards, plus 4 K512 affine diagnostic arms x 240 = 960 logical affine
steps and 3,840 backwards: total 2,400 logical optimizer steps and 9,600 microbatch
backwards.  Every aligned/reversed/affine logical step computes one target prefix and
one native teacher suffix; a document-deranged score step additionally computes the
donor prefix.  The executable transaction must receipt these phases separately and may
not silently vectorize, retain/cache teacher logits across steps, call Adam per
microbatch, or change optimizer-step count.  Affine folding has zero marginal deployed
cost for one chosen program, but retaining both program variants and fitting 960 steps
has explicit artifact and experimental cost.

Including the frozen reporting sweep, the complete physical census is 2,940 prefixes;
2,460 teacher suffixes; 10,680 student suffixes; 13,140 total suffix returns; 2,400
optimizer steps; and 9,600 two-row backwards.  Attention and native MLP sites 0--3 are
called 2,940 times each; attention and MLP sites 4--17 are called 13,140 times each.
The student native-MLP3 call count remains exactly zero.
