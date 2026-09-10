# One shared source-scoring rule for the has/had and is/was paths

The established MLP4→L9H1/H4 paths share contextual reading structure but not a
common sufficient value direction. Test a computational hypothesis: these heads
can use fixed Q/Q2 queries shared across the two tasks, leaving contextual keys
and values to carry their differing information. This is a source-dependent
router, not a claim of constant attention patterns or shared payloads.

Use the original unfiltered 96 paired rows of the has/had and is/was shared-path
study. Its direction-stratified A1 fit halves contain16 has and8 is pairs.
Evaluation contains has A1/A2=16/32 and is A1/A2=8/16 pairs. Every native error
is retained. No capability filtering is used. Exact token-sequence overlap of
fit with each evaluation panel is zero. All texts have been opened historically;
A2 is a construction shift relative to this fit, not pristine new OOD evidence.
The saved manifest records a legacy executor hash drift limited to the shared
producer module. Each task row builder still validates; the modern backend and
checkpoint are separately bound. No old full-executor replay is claimed.

For each head h∈{1,4} at layer9 and each Q kind, average raw pre-head-RMS query
projections at the final prediction position across BOTH endpoints of all fit
rows within each task. The candidate prototype is the equal-weight mean of the
two task means, irrespective of differing sample counts. Store as native FP32.
Do not optimize, normalize, select examples, tune coefficients or sweep heads.
The native head RMS and actual stored RoPE tables remain live after replacement.

At each evaluation endpoint compare four arms: native, shared fixed queries,
own-task fixed queries (diagnostic), and native queries with both selected head
outputs zeroed at that endpoint. All other tokens/heads and the full suffix are
native. Four fit forwards plus32 evaluation forwards =36/624 sequences. Reuse
one model and existing batch/capture/scoring machinery.

The actual fixed-query routing computation can fold into the key weights:
score_i(t,s)=x_sᵀ W_Kiᵀ R_sᵀ R_t norm(qbar_i) /
sqrt(||W_Ki x_s||²/128+epsilon), with pattern=score_1*score_2/128².
x_s is the already residual-normalized source input. The key norm cannot be
dropped. Use actual native stored rotary coefficients; BF16 sine/cosine tables
need not be exactly orthogonal, so do not replace their transpose by an inverse
or assume an exact relative-phase identity. Payload, value mixing and output
maps remain required. No full1152³ tensor is materialized.

A: source/row/checkpoint hashes; counts36/624; finiteness; restored hooks and
read-only capture wrappers; native inputs unchanged at the edited attention;
query-prototype edits confined to selected heads/endpoints; folded-pattern
versus actual native pattern maxabs<=1e-3 AND relative Frobenius<=1e-5, on
every valid source position for shared and task-prototype arms. CPU controls
check untouched positions, restoration, nonorthogonal rotary folding, and a
counterexample showing key normalization matters. There are no weight updates.

B: shared-prototype endpoint distributions match native on EACH of the four
evaluation panels: mean teacher KL<=.001, p99<=.01 nats, zero top1 changes.
Report own-task prototype results under the same bars but do not adopt it as
a fallback after shared failure.

C: shared-prototype predicts native base→donor full centered-logit change with
relative RMS error<=.10 on EACH panel, and donor-answer-minus-foil margin-contrast
relative error<=.10. Reference norms<=1e-8 use absolute1e-8 error. This is a
basic screen; it is not the final full-program fidelity/adoption standard.

D: selected-head endpoint removal is live on EACH panel: pooled centered-logit
removal effect norm is >=.01 times the native paired-effect norm (and >1e-8).
This prevents a fidelity pass on an irrelevant interface. It does not require
these two heads to contain the entire task. No new selective removal/composition
claim is made here; those are required after a passing basic candidate.

All gates must pass to continue this shared-query candidate. If task prototypes
also fail, fixed queries are insufficient at this boundary; if only sharing fails,
task-conditioned queries are descriptive evidence, not a promoted replacement.
No new rank, head subset, mixture, dose or prototype-family search follows by default.

Literal price: shared prototypes512 scalars versus1024 for the two task libraries.
A standalone selected-head endpoint read would avoid four128×1152 query maps
(589824 scalar coefficients/MACs) but still needs key norms, values and prefix.
The hook implementation saves no stored native weights or executed projections;
all545902902 native parameters remain charged. A successful rule could simplify
the local circuit description but is not a whole-model saving or four-property
completion. Managed GPU only, alarm600s; no gradients or model training.
