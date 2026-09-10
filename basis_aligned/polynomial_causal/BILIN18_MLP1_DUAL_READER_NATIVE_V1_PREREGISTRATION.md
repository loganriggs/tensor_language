# Native consequence of a two-task constrained MLP1 output split

The saved-reader CPU screen passes: independent dual writes require at most
1.056784 times each task's unconstrained minimum write norm. This makes a
native selectivity test eligible, but does not establish a circuit. Test only
MLP1 and the two unchanged rank4 task readers. No new rank/site/head selection.

Restore temporal-v12/iswas-v11 target rows exactly as compiled_response_ood:
12 capable rows per A1/A2 panel per task, using the frozen capability artifacts;
retain all selected rows. Controls are its existing first16 temporal-v12 P rows.
These texts were opened in earlier studies; call this held from reader fitting,
not new pristine OOD evidence. Iswas-specific control families remain untested.
Reuse the existing row builders, backend, capture and final-margin/logit scorer.
Before any model work check equal per-row base/donor lengths and aligned semantic
positions; fail rather than repairing or filtering an invalid row.

Capture native base and donor MLP1 outputs and final states for target and control
cohorts (four forwards). For each cohort run exactly seven arms:
zero replay; ordinary temporal projection; ordinary iswas projection;
dual temporal; dual iswas; sum of both dual projections; orthogonal joint-span
projection. Each installs a donor-minus-live-output component at MLP1 through
all positions up to the row's semantic endpoint, with the native suffix fully
recomputed. Total18 forwards, no fit/update. Independent single-task output
projectors are C_t^T(C_t C_t^T)^-1 C_t; dual projectors are D_t C_t with
D=C^T(CC^T)^-1. Their sum equals the orthogonal joint reader-span projector.
No response addition shortcut for final logits.

A: frozen source/prior/reader hashes, row and count checks, finite states,
native zero replay and dual-union versus orthogonal-union full-vocabulary
logits abs<=1e-3 AND relative<=1e-5; local dual readout, other-task preservation,
sequential/joint edits and hook restoration. Local FP64 algebra abs<=1e-9;
deployed FP32 readout checks abs<=1e-3/relative<=1e-5 for nonzero targets and
abs<=1e-3 for the preserved opposite-task change. Keep all native errors.

B: both task populations retain at least90% of their ordinary own-task signed
margin effect under the dual own-task edit. Define signed effect by projection
of the intervention's donor-answer-minus-donor-foil margin change onto the
native donor-minus-base margin-change vector. Each ordinary own-task reference
must explain at least1% of that native vector (positive sign); otherwise B fails
as an insufficient local signal, not an invalid numerical instrument.

C: each dual cross-task margin-change RMS is at most10% of its dual own-task
margin-change RMS on the same target population. For both single-task edits,
temporal-P mean teacher KL is no greater than the corresponding ordinary edit
plus1e-6 nats/token, with zero native top1 flips. Report all signed projections,
RMS values, KL and flips, even when B fails. Fixed bars; no row/mode/dose rescue.
These are basic-screen bars, not extraction or adoption thresholds.

Also report native output interactions for the two dual edits. Their local
operators commute, but nonlinear suffix effects need not add. No additive
final-effect assumption or semantic sharing claim follows from the local split.
If B and C pass, a later independently registered test must address reverse
interchange, task-specific controls, fresh OOD, actual removal and an independently
executable reduced computation. If they fail, preserve the local algebra and
reject promotion; do not expand to neighboring sites automatically.

Managed GPU only, alarm600s, native weights unchanged and all545902902 parameters
plus both9216-scalar reader/writer adapters charged. Use the small shared
intervention primitive with existing capture/scoring, not another executor stack.

The zero-forward ROW_AUDIT receipt now fixes target/control row hashes and
confirms all alignments plus zero target token-sequence overlap with reader
fitting. Match those hashes before native execution. Cache each unchanged row
builder within the process, or serialize its rows once and hash-bind the manifest,
to avoid repeatedly reconstructing identical populations during preflight.
