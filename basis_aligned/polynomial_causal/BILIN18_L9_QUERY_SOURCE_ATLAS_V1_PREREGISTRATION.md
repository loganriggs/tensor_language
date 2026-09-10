# L9 query-source atlas v1 — registered before native source capture

2026-09-10 02:04 UTC. Original handoff/pilot joint read–route–write target.
The fixed-query and direct-embedding source hypotheses failed. Identify the
upstream query dependencies of the shared L9H1/H4 consumers, across has/had
and is/was. This is an exploratory localization screen, not circuit adoption.

The 19 sources are the direct embedding (including every skip/reentry
coefficient), followed by attention and MLP writes at layers 0 through 8,
transported by the exact-real product of subsequent residual coefficients.
Source index 18 is MLP8. Source values are held fixed at the native forward.
Edits change the query-input edges of both heads at the semantic endpoint;
they do not delete an upstream module or regenerate other source terms.
Keys, mixed values, all other heads/tokens and the full suffix remain native.

Use the unchanged manifest BILIN18_L9_SHARED_QUERY_ROUTER_V1_ROWS.json:
72 eval pairs, four panels, all native errors retained, no fit. These texts
have already been opened; neither panel is a fresh validation set after this
atlas. Prior source-position weight studies examine a different branch; they
do not supply this 19-source normalized query-edge factorial.

For each panel/endpoint, capture sources once. Compile source_gain_attention:
each head read has a quadratic numerator in 19 source gains, divided by the
square root of two positive quadratic norm factors. Preserve actual rounded
RoPE, RMS epsilon, causal mask and all Gram cross terms. Replace only the
selected head reads immediately before the native output projection.

Arms: native; unity (all gains 1); zero; each of 19 singleton masks; each
of 19 leave-one-out masks; direct_omit_mlp8 (independent raw query-port
implementation of the last leave-one-out mask). 42 arms, both endpoints,
four panels = 336 full forwards, 6048 sequence evaluations. No model updates.

A instrument: bound sources/checkpoint/rows; CPU FP64 controls <=1e-11;
every gain's tensor read agrees with a separate nested-RMS/query-dot direct
oracle at max absolute <=1e-8 AND relative Frobenius <=1e-9. Native unity
head read, normalized source sum, full unity logits and independent raw
MLP8-omission full logits agree at <=1e-3 AND <=1e-5. Every input finite;
capture and intervention hooks restored; exact forward counts. Native paired
margins replay the parent. Gram norm must be nonnegative without clipping.

B simple shared source hypothesis: at least one SAME singleton passes all
four panels: native KL mean <=.001, p99 <=.01, zero top1 changes, paired
centered full-logit and answer-margin contrast relative errors <=.10
(absolute <=1e-8 for zero targets). Report every singleton and common set.
Passing B only nominates a source for genuinely fresh confirmation; no
multiple-testing significance or independent identification is claimed.

C shared dependency hypothesis: define a task's robust necessary-source set
as sources whose omission changes centered endpoint logits by >=.10 times
the native paired-effect norm in BOTH that task's panels. Both task sets
must be nonempty and their Jaccard overlap >=.75. This is an operational
dependency comparison, not semantic equivalence. Save all individual scores.

D live consumers: zero-query read changes centered endpoint logits by >=.01
times native paired-effect norm and >1e-8 in every panel. A failure invalidates
the instrument; B/C failure preserves the corresponding null. No source,
rank, dose, head, position or norm-denominator rescue after outcomes.

Literal price: full 545902902 native parameters remain; actual weight and
projection savings zero. Per endpoint transient source bank19*1152, query
readers4*128*1152, native keys/values, source Gram and projected coefficients
remain charged. A degree-two source polynomial has190 monomials. Retain
the factorized contraction when cheaper than materializing190*128 entries
per head. Prefix capture and full suffix are not independent extraction.
600-second watchdog; GPU only through managed enqueue. Save immutable JSON.
