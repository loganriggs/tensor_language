# Does the direct embedding suffice for the shared heads' query input?

The fixed-query screen is a valid null. Task-specific constants also fail on
three of four distribution panels. This next candidate is an explicit producer
deletion from the original handoff's repeated-embedding architecture, not a fit
of more prototypes, a rank increase or a selected head subset.

Let e_t=RMS(Wte[token_t]) be the original normalized embedding. In exact real
arithmetic, the coefficient of its direct skip/reentry contribution before
attention9 is alpha, obtained from alpha=1 then alpha=lambda[l,0]*alpha+
lambda[l,1] for l=0,...,9. Attention/MLP-generated terms are excluded from this
direct lineage even if they also depend on the token. The remaining contextual
terms are not asserted independent of e. No complete semantic field is claimed.

At the final prediction position of L9H1/H4, replace ONLY the Q and Q2 projection
inputs by RMS(alpha*e_t). Compute the two projections using their actual weight
rows, then let native head normalization and actual RoPE execute. K, K2, mixed
values, other heads/positions, residual and complete suffix remain native. The
query normalizer is recomputed from the selected source; do not retain the old
contextual RMS denominator. alpha=0 makes this candidate ineligible; no rescue.

This is a token-derived query shared by both tasks, with no row-dependent fitted
constant and no native contextual query-state input. The existing weight maps
remain opaque and charged. Query-only edits are a declared native port API, not
an edit of the entire attention input or an independent claim of task erasure.

Reuse all72 evaluation pairs in the fixed-query manifest: has A1/A2=16/32 and
is A1/A2=8/16. No fitting, filtering or new text. These texts are opened; this is
a changed intervention hypothesis, not pristine OOD confirmation. Each endpoint
has native, embedding-query, identity-query replay and native two-head removal:
32 forwards/576 sequence evaluations. Fit rows are unused.

A: source/row/checkpoint hashes and counts; finite nonzero alpha; hook restoration;
unchanged pre-attention inputs and unchanged nonselected query outputs;
identity-query full-logit replay maxabs<=1e-3 AND relative<=1e-5; native margin
contrasts replay the fixed-query parent within1e-3/1e-5. Independently compare
selected native projection outputs with FP64 weight contraction of the chosen
normalized source within1e-3/1e-5. CPU controls verify recurrence coefficients,
query-only scope, all unselected positions, restoration and that query-specific
normalization is live. The stored model is never edited.

B/C/D are unchanged from the parent screen, now for the embedding-query arm:
EACH panel mean native teacher KL<=.001,p99<=.01,zero top1changes; native paired
centered full-logit and answer/foil contrast relative errors<=.10 (zero norm uses
absolute1e-8); native both-head removal is live with endpoint effect norm>=.01
native paired-effect norm and >1e-8. All three plus A must pass for continuation.
The identity and removal arms are controls, not alternative candidates.

A failure rejects removing contextual query input at this fixed boundary. It
does not justify a nearby-layer/head scan or a fitted embedding-plus-offset
repair. A pass still needs a fresh task population, independent circuit execution,
selective producer/consumer interventions, joint composition and literal structural
reduction before promotion. All545902902 native parameters remain; no actual
projection/storage savings in this hook test. No learned query prototype exists.
Managed GPU only, alarm600s; no gradients or persistent model updates.
