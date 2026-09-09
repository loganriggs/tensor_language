# Shared join-kernel v2: one calibration-coverage repair

V1 is preserved as instrument-invalid: all held panels contain unmatched calibration
role/lag cells (14 IID,16 OOD,9 fixed point); no held model output was evaluated.
Its executor controls pass and physical count is361776, but these establish no
trained fidelity. The printed v1 exact error0 is an empty maximum, not closure.
The fitted matched-score RMS residual .7838 is a calibration diagnostic only.

One repair is authorized by the instrument-failure protocol, before new held outcomes:
retain the original256 calibration inputs15909 and append96 query forks of one
new independent IID24-cycle world16909, crossing all24 entities with all4 hops.
This guarantees query role/lag support independent of held data. All352 examples
enter the same matched-score fit, including their binding positions. The repeated
calibration prefix is explicitly counted, not claimed as96 independent worlds.
No new record fields, coefficients, heads, ranks, lag features, ALS iterations, or
fidelity thresholds. Coefficients are refitted once with the same8 fixed passes.
No choosing among fitted candidates. Original v1 remains an invalid instrument.

New held seeds16910/16911/16912 replace15910/15911/15912; sizes/topologies unchanged.
No held tokens or outputs are constructed before the new coefficients are saved.
A synthetic support audit enumerates structurally possible binding-lag groups and
query groups; it must pass before GPU enqueue. Coverage repair is the only change.
All A/B/C/D gates, exact controls, prices, failure closure and runtime caps below
remain binding. If this now-valid test fails, close this fixed local-record kernel.

## Original frozen semantic hypothesis and gates (v1 reference)

# Shared semantic join-kernel replacement v1

The frozen L2H1/H2 middle-label test passes all4 gates. Single-column mismatch
reduces joint score RMS to2.7–6.9% and causal output RMS to0.9–4.3% of matched;
renaming both columns restores a live route. The matched gate uses the complete
product-attention score, not guessed meanings for individual QK factors. Matched
score energy lies overwhelmingly at binding-value→binding-value cells (98.7–99.7%);
other cells are retained in all reports and are not declared algebraically zero.

Now test one actual replacement of both heads' entire L2 query/key score computation
on cold inputs (up to51 tokens:24 bindings followed by Q/entity/hop). This scope
includes every binding/input position, not just the scored answer. Longer histories
are unsupported by this first candidate and must not silently enter its evaluation.

Define causal local records from tokens, with no function labels or model activations:
at a binding-key token u, the partial record is(u,u); at its following value v, it
is(u,v). Q has no entity record. The query-entity token e and following hop token
both have record(e,e), with distinct role labels. Eight roles are binding key/value,
Q, query entity, and four hop markers. This is a new fixed extrapolation of the
confirmed value-position mechanism to other native destinations, to be falsified.

One equality predicate is reused with swapped fields: H1 compares destination value
with source key; H2 compares destination key with source value. Predict the complete
native product-attention score as

  A_h(t,s)=1[fields match] * g_t^2*g_s^2 * gamma[h,middle]
             * theta[h,query_role,source_role,t-s] * 1[s<=t] * distinct_binding(t,s).

When both positions belong to initial bindings, distinct_binding excludes the
same binding pair, including self. This follows the proposed distinct-edge join
and makes its fixed-point behavior explicit; no unobserved same-edge coefficient
is silently inferred from non-loop training. Query-record reads are not subject
to that binding-pair exclusion. This entire extrapolation is fixed before fitting.

g is the original live L2 input RMS gain, already shared with retained readers;
no cached teacher state is used in execution. Theta includes signed positional
coefficients; gamma is positive with mean1 separately per head. Native V/O, other
L2 heads and the full prefix/suffix remain live and charged. Physically remove
H1/H2 rows from each Q1/K1/Q2/K2 weight matrix (32768 constants); native heads0/3
keep their original rows and exact absolute RoPE. Candidate position coefficients
are an explicitly approximate fitted rule, not an exact lag-RoPE identity.

Fit on256 fresh IID24-cycle cold inputs, seed15909, with random queryhop0..3.
Use only native L2 scores on matching field pairs, with g_t²*g_s² as the regression
feature. Eight fixed weighted alternating-least-squares passes fit theta and gamma;
gamma clamp>=1e-6 and per-head mean normalization are fixed. Freeze/save/hash before
held outputs. No head/rank/feature/iteration sweep or output fitting. Unseen matched
role/lag cells in evaluation are instrument-invalid, never silently defaulted.

Held panels:64 IID cycles seed15910,64 two12-cycle worlds seed15911,32 fixed-point
query worlds seed15912. Random binding order and random queryhop0..3 for positive
panels; fixed-point controls use hop3. Retain native errors: target is full teacher
distribution, not agreement with the task algorithm. Test full, L2H1 removed,
L2H2 removed, both removed, by zeroing the corresponding complete head pattern in
both native and candidate execution. All positions and all29 logits are scored.

A instrument: native retained-head/scatter replay, planted nonzero semantic-kernel
and fitting controls, causality, all evaluated matched cells seen, finite values.
B full-distribution fidelity: native-to-candidate KL mean<=1e-3,p99<=1e-2 on every
population/arm, all positions and final-query positions separately.
C removal/composition: centered full-logit intervention vectors relative RMS error
<=.01, or absolute<=1e-8 if native RMS<1e-6, on both singles and joint, each panel.
D structural reduction: removed Q/K rows physically absent; all6528 theta and48
gamma constants charged, including unused cells. With the known generic final-readout
fold, expected total361776 versus387968 for its native-background baseline. The
12672 final-fold saving remains generic; new semantic savings would be26192.

Failure closes this fixed local-record kernel; no posthoc extra record fields or
lag/role/head sweep. A pass licenses portable export and a fresh joint field-edit
test, not full bilin18 completion. No native or learned value/readout coefficient
is hidden in an unpriced adapter. Batch4 FP64,1800s,<256MiB/analysis tensor, GPU only
via managed enqueue. Exact original-model replay is a separate control from the
approximate fitted kernel; never relax the former to rescue the latter.
