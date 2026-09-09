# Token-payload reuse by the licensed four-head circuit: basic screen

The bilinear reconstruction handoff and appended structural criterion control.
After the small-model query block/square/single-gate nulls, test an explicit
reuse opportunity in the actual18-layer model. No fitting or rank selection.

The previously licensed dual-command union is L9H1,L9H4,L11H3,L15H5 (zero-based).
Its attention values are (1-lambda_l)*V_l(RMS(u_l))+lambda_l*v0. Here v0 is
the shared first-layer token-value stream, with the same head indexing across
layers. Lambda is unconstrained, not a convex weight. Candidate: these four
heads can read token-only v0, removing their contextual value projections.
The input to v0, query/key routing, native normalizers and all downstream
states remain explicit native computations, recomputed for each arm.

Prior art checked: architecture contract; raw-factor pilot; exact shared
head union; aspectual/tense mixed-value subspace null; bank-routing/local-value
factorial. Those factor studies changed upstream states under frozen token
inputs or fit mixed-value geometry. This is a direct end-to-end deletion of
the selected contextual value-producing weight rows, at every position.
It is not a donor-head clamp or subtraction from a frozen native suffix.

Use all32 registered dual-command-v2 rows x four endpoint cells =128 sequences,
preserving FIT/HOLDOUT labels and every template. These are already opened
texts, not fresh/OOD evidence. Two independent command outputs (will/had and
is/was) share the same sequence. Exact existing builder and licensed parent
result are hash-bound before model access. No row filtering or new head choice.

Arms: native; A deletes L9H1/H4 contextual values; B deletes L11H3/L15H5
contextual values; AB deletes allfour. Install zero c_v output slices, with
layer0 and v0 unmodified. The all-four candidate is the only nominee; A/B
diagnose its composition and cannot be adopted as a fallback after failure.

A instrument: deterministic complete rows, parent license and source hashes;
hook selection/restoration; fixed head/shape config; finite outputs; all arms
are exercised; exact forward/sequence count. Tiny FP64 controls independently
check selected heads equal native QK reads of lambda*v0, unselected heads
remain equal within the same attention call, negative/>1 lambdas, live raw
removal, and joint hook selection. Native full-token output is evaluated
through the already used manual forward. No new full model executor.

B candidate fidelity: for AB, in EACH phase (FIT/HOLDOUT), all-token teacher
KL mean<=1e-3 and99th percentile<=1e-2 nats/token; report maximum and per-template
metrics. Exclude padding, not native errors. At each of the two command
positions require native-argmax agreement>=.99 in each phase, and preserve
the paired command signed-logit-change vectors with relative L2 error<=.01
(denominator floor1e-6; report raw denominator). Compute the paired difference
within each four-cell row, not gold-token accuracy as a fidelity substitute.
Report native and AB task accuracy separately, using the cell's actual target.

C composition: all-token centered-logit interaction
AB-A-B+native has RMS<=.01 times AB-native RMS in each phase, with a fixed
1e-6 denominator floor. Report raw effect/interaction norms and the fraction
of zero native effects. This is a diagnostic necessary gate for an additive
composition explanation; it does not assume logits/probabilities add.
The two groups' actual joint execution is always measured.

Only A+B+C licenses fresh/OOD replacement and explicit extraction work.
Otherwise preserve the null and stop the token-only payload nomination;
do not rescale lambda, select heads/tasks/templates, or weaken the bars.
Even a screen pass is not the user's full structural success condition.

Price: one checkpoint load;32 batches of four sequences;four arms each,
128 transformer forwards/512 sequence evaluations. Full logits are evaluated
but reduced to scalar row/token metrics before the next batch. Maximum
logit tensor4*27*50304*4=21731328 bytes, below256MiB. This cap concerns
new analysis tensors, not resident checkpoint tensors. No gradients/training.
Run only through managed bqrunner, GPU alarm600s. The deployment candidate
could omit4*128*1152=589824 contextual value weights; the hook prototype
still stores/evaluates them and claims no realized saving. Native QK, v0
producer, adapters and all remaining weights remain charged. This is a
mechanistic dependency screen before any reduced executable is built.
