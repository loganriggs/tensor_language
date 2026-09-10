# Two task readers of the attention-times-pre-attention bilinear computation

The dual-reader native screen is valid and retains own-task effects, with small
cross-task effects, but FAILS the registered P no-worse predicate. It is not
promoted. Do not retune that predicate, fit another reader, or scan another site.
The next question is computational: which exact numerator operation supplies
the two task-specific MLP1 writes?

Fixed native streams: u is the layer1 pre-attention residual after original-
embedding injection; a is the complete new attention1 residual write. Neither
is named a pure token or context feature. The MLP sees n=(u+a)/s, where
s²=||u+a||²/1152+native_FP32_epsilon. For the same eight reader covectors C,
fold native Down into E=C Down and compute

pre = E[(L u)*(R u)]/s²,
cross = E[(L u)*(R a)+(L a)*(R u)]/s²,
attention = E[(L a)*(R a)]/s².

Their sum plus C*b is the reader output. This is an expanded computational
program with one shared denominator, not three independently normalized paths.
A component edit holds the full world-specific norm contribution present.
It is an explicit output-node edit, NOT an attention-input knockout, independent
physical-source removal, or path-independent allocation of norm changes.

Prior art checked: rungs487–491 polarized MLP0-branch changes and named the
native-state source, with attention1 necessary for T/I; rung492 did not establish
a portable attention1->MLP1 path. The September2 math review caught the invalid
use of raw attention differences as normalized MLP inputs. Rung507 already
provided general source-pair machinery at MLP10. This follow-up instead fixes
two task readers and the complete u/a native numerator at MLP1, explicitly
retains the shared norm, and asks whether the same operation supplies both
task-typed effects and resolves the current P collateral. It does not repeat
or promote the failed reader-path claim.

Use the exact saved48target+16control row manifest from the dual native parent.
Capture native base/donor streams during four forwards by read-only hooks.
Reconstruct u from normalized embedding, actual attention0/MLP0 outputs and
native lambdas; audit against the actual MLP1 normalized input. No new forward
implementation. Retain native weights, bias, epsilon and the original suffix.

On each target row use its own task's existing dual output writer; on P controls
measure temporal and iswas writers separately in a concatenated32-row control
cohort (the same16rows repeated with distinct edit labels; no population
selection). Five patch arms: full native dual reader interchange; pre-only
component interchange; cross-only; attention-only; sum of compiled components.
Four native source forwards (target48+control16 each base/donor), ten patched
forwards (five target48 plus five control32) =14forwards,528sequence evaluations.
The duplicated controls require distinct batch row IDs while retaining the
original source-row IDs for scoring and provenance.

A: frozen hashes/counts; exact component algebra FP64 abs<=1e-8/relative<=1e-9;
normalized-input native replay abs<=1e-4/relative<=1e-6; compiled reader native
FP32 abs<=1e-3/relative<=1e-5; summed-component versus full-dual full-vocabulary
logits abs<=1e-3/relative<=1e-5; parent per-task own-effect/P-control replay under
the same tolerances; hooks restored and nonzero component controls.

B: cross-only signed effect is >=90% of the full-dual signed native contrast
projection on BOTH task populations. Reference must be positive and>=.01 of
the native contrast. Also require per-task cross-only margin-effect vector
relative error<=.15 versus the full-dual margin-effect vector. Other component
results are descriptive, not fallback candidates selected after the result.

C: for BOTH labeled P edits, cross-only mean teacher KL<=the original ordinary
(single-reader) parent mean KL+1e-6, and zero top1 flips. This retains the failed
comparison's original control bar for the new computational candidate. No
posthoc dose/coordinate/subset repair. If B/C fail, preserve the operation-level
result and do not promote another component from its descriptive profile.

No new fit, training or weight update. Managed GPU, alarm600s. Report native
parameters545902902, reader/writer adapters, folded maps and extra local
contractions. No structural saving or independent extraction is claimed from
retaining native background or evaluating an algebraic identity. A pass would
license reverse/removal/fresh-OOD/joint-task tests of this specific operation,
with all remaining opaque coefficients charged.

## Pre-execution batching clarification — September10, before native integration

Keep each P edit at the parent's16-row batch geometry. Run the temporal and
iswas control arms as separate labeled batches, reusing the same native source
captures. This replaces the proposed concatenated32-row patch batch and its
artificial row-ID duplication. It avoids introducing a new FP32 GEMM geometry
into the strict parent-KL replay comparison and directly reuses the parent
intervention/scorer. No candidate outcomes have been opened, and no scientific
bar, population, edit or selection changes. Correct price: four source forwards
plus five48-row target forwards and ten16-row labeled-control forwards =
**19 forwards and528 sequence evaluations**. All earlier14-forward/duplicated-ID
instructions in this protocol are superseded by this clarification.
