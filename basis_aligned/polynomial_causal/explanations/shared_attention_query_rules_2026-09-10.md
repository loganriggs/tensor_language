# Can the same attention heads use a simpler shared query rule?

September 10, 2026, 01:45 UTC.

Two concrete query simplifications have now been tested on the trained 18-layer
model. **Neither meets the registered fidelity criteria.** A shared fixed query
works on one small is/was panel but fails across the full evaluation. Replacing
the query's contextual input with its direct token-embedding source fails more
strongly. The implementations and weight-folding checks pass, so these are
scientific nulls rather than execution failures.

This follows the original [bilinear handoff](bilinear_circuit_reconstruction_codex_handoff.md)
and its [pilot](bilinear_reconstruction_pilot_report.md): investigate an explicit
shared read–route–write computation, retain normalization, and test its causal
behavior. The present tasks are **has/had and is/was**, whose contextual paths
share L9H1/H4. They differ from the recent will/had versus is/was MLP1 cohort.

## Why test the query side?

Earlier path experiments support the same attention-head locations and similar
contextual-value operations for these tasks, but their sufficient value directions
are task-specific. Co-use of a module is therefore not evidence of one shared
semantic variable. The new hypothesis asks whether its *source-selection rule*
is reusable while the carried values differ.

For each head, the model multiplies two normalized query–key dot products and
uses the result to weight a value. It has no softmax in this product attention.
A fixed query would turn the head into a contextual source scorer, without a
context-dependent query input at the final prediction position.

## Test 1: fixed queries shared across tasks

For each selected head and each of its two query projections, average the raw
native query vectors across both endpoints of the fitting pairs. Average the
two task means with equal task weight, then freeze the FP32 result. This gives
512 shared constants. Separate task means use 1,024 constants and are diagnostic
comparators; they are not fallback candidates to promote after a failure.

The original unfiltered population has 96 pairs: 24 fit and 72 evaluation.
Evaluation has four panels: has A1/A2 with 16/32 pairs and is A1/A2 with 8/16.
No evaluation token sequence occurs in the fit set. Native mistakes are retained.
All texts were previously opened; A2 is a construction change relative to this
fit, not pristine new OOD confirmation.

At both endpoints, compare native execution, the shared query, the own-task
query, and removal of both selected head outputs. Only the selected heads at the
prediction position change. Keys, values, other positions and the complete suffix
remain native. The removal control confirms the interface matters on every panel.

| Panel | Shared-query mean KL | Paired full-logit effect error | Prediction changes |
|---|---:|---:|---:|
| Has/had A1 held half | .011315 | 17.33% | 2 |
| Has/had A2 | .007206 | 13.21% | 3 |
| Is/was A1 held half | .000876 | 5.53% | 0 |
| Is/was A2 | .003011 | 9.94% | 2 |

KL compares the entire candidate distribution with the native distribution.
Paired effect compares the change in centered vocabulary logits between base and
donor contexts. The gates require mean KL≤.001, p99≤.01, zero prediction changes,
and ≤10% error in both the full paired effect and the answer/foil margin contrast,
separately on every panel. Only is A1 passes all these basic conditions.

Task-specific means also fail distribution fidelity on three panels. Pooling is
therefore not the only problem with this construction. The fitted task means have
direction cosines .824–.949 across the four query/head combinations. Switching
between pooled and task means changes paired margins by 3–6% of native contrast.
These are descriptive comparisons, not evidence for an optimal constant-query
program or an impossibility theorem about all fixed queries.

The [native receipt](../BILIN18_L9_SHARED_QUERY_ROUTER_V1_RESULT.json) is valid:
36 forwards/624 sequence evaluations, 2.284 seconds, no gradient updates.

## The weight fold is correct, including normalization and rounded RoPE

Let x_s be a source's residual-normalized input, W_K its key projection, and qbar
the fixed query after native head normalization. With actual stored rotary maps,

    score(t,s) = x_sᵀ W_Kᵀ R_sᵀ R_t qbar
                 / sqrt(||W_K x_s||²/128 + epsilon).

The two such scores multiply, with the native factor 1/128². The query can be
folded into the source's linear reader; the key norm still requires its own
computation. Positional dependence and payload production do not disappear.

The native sine/cosine tables are rounded to BF16. Their transpose remains valid
in this identity, but it is not generally their exact inverse. Nor may we assume
an exact relative-phase identity between rounded tables. The implementation uses
the actual stored coefficients.

Folded scores match the native patterns to maximum absolute error 2.69e−7 and
relative Frobenius error 2.49e−7. Six CPU controls pass, including deliberately
nonorthogonal rotary coefficients and a live key-normalization check. This verifies
the algebra; it does not rescue the failed behavioral hypothesis.

## Test 2: a query computed from the direct embedding source

A fixed average throws away token identity. A different, weight-derived proposal
uses the model's explicit original-embedding pathway rather than fitting more
prototypes. Write e=RMS(Wte[token]). Its direct coefficient before attention9 is
obtained from

    alpha = 1
    for layer in 0,...,9:
        alpha = lambda[layer,0]*alpha + lambda[layer,1].

The actual coefficient is **34.5760605212**. This number does not measure causal
importance: the other computed writes can reinforce or cancel that direct term.

The candidate feeds RMS(alpha*e) into Q/Q2 only, then executes native head norms
and RoPE. It recomputes the selected source's norm instead of borrowing the
contextual query denominator. K/K2/V, other heads and residual paths remain live.
This retains the direct skip/reentry embedding source, not every computation that
depends on the token. No complete semantic token field is claimed.

| Panel | Embedding-query mean KL | Paired full-logit effect error | Prediction changes |
|---|---:|---:|---:|
| Has/had A1 held half | .046118 | 27.05% | 12 |
| Has/had A2 | .045251 | 27.28% | 15 |
| Is/was A1 held half | .025510 | 27.13% | 2 |
| Is/was A2 | .032660 | 30.43% | 5 |

The [second receipt](../BILIN18_L9_EMBEDDING_QUERY_V1_RESULT.json) is also valid:
32 forwards/576 sequences, 2.180 seconds, no fitting. Identity-query replay has
maximum full-logit error 1.53e−5; the FP64 projection oracle misses by at most
6.42e−6. Native paired margins and the earlier removal metrics replay exactly.
The reused scorer is checked against that completed parent result.

**Scope clarification:** this rejects the specified normalized direct-embedding
replacement. It does not prove that every token-only nonlinear query program is
impossible. The machine terminal “contextual query input required” must be read
within this registered intervention family, not as a universal representability
claim. No fitted offset, nearby head or weaker dose is adopted.

## The next executable mathematical object: contextual query sources

Before attention9, the residual input has the exact-real source organization

    u = alpha*e + Σ[j=0..8] gamma_j*(attention_write_j + MLP_write_j),
    gamma_j = product[k=j+1..9] lambda[k,0].

These are 19 source terms. For an edit of these *consumer input edges*, hold the
upstream source bank fixed and assign gains z_s. This differs from deleting an
upstream module globally, which would change later sources too.

Stack the four Q/Q2 projection maps for H1/H4 into W. For each source f_s, retain
p_s=W f_s and the source Gram G_st=f_sᵀf_t. Then

    p(z) = Σ_s z_s p_s
    rho(z) = zᵀ G z
    normalized_query_head(z)
      = p_head(z) / sqrt(mean(p_head(z)²)
                        + epsilon_head*(rho(z)/1152 + epsilon_residual)).

The last expression exactly combines residual RMS, linear projection and head
RMS over real arithmetic. It preserves the cross-source norm terms needed for
joint edits. Individually normalizing sources or dropping off-diagonal Gram
entries would define a different computation. RoPE follows this query computation.

[Five CPU controls](../PROJECTED_QUERY_SOURCE_EDITS_V1_CONTROLS.json) now pass
single, joint, signed and zero-source edits, with query error 5.55e−16. The native
lambda coefficients reproduce the embedding coefficient above. One planted norm
case was strengthened before any native source-bank evaluation to make omission
of cross-Gram terms visibly fail; the control receipt records that change.

This is preparation for native source attribution, not an identified shared
producer. No native 19-source bank has yet been captured. Initialization still
needs native states/weights; Gram cancellation can make a finite-precision norm
evaluation unreliable. A packed source-edit state would use 9,918 scalars per
endpoint; the current dense Gram layout would use 10,089. Both exceed one original
1,152-coordinate residual because they retain separately editable sources.

## Status against the actual goal

Neither tested simplification supplies a new circuit satisfying the four properties.
All 545,902,902 native parameters remain charged; the hook tests realize no weight
or projection savings. The useful progress is narrower: we tested explicit shared
routing rules, verified their exact weight folds, preserved their failures, and
implemented a normalization-aware way to investigate the contextual producers that
the next circuit explanation must account for. Another prototype or rank sweep is
not the next scientific step.
