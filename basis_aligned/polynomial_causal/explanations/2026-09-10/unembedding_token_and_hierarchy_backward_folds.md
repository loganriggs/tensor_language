# Backward unembedding folds: individual tokens and shared structure

**Both requested views have been run.** The exact token-reader program folded through MLP17 and MLP16 predicts the tested live vocabulary effects with 4.6–5.6% relative error. A 16-leaf hierarchy built from unembedding weights does not replace the individual readers: its prediction error is 94–98%. The token-specific remainder is still essential under this test.

This is a conditional prediction result. The MLP16 swap being predicted slightly opposes the correlative task; we have not extracted the task circuit, identified a generally sufficient hierarchy, or reduced the model’s structural description.

## The two views

**Individual tokens:** each vocabulary row U_t is a direction that scores a token from the final residual vector. We explicitly contracted 518 readers: every 98th vocabulary row plus the answer/foil IDs of the frozen tasks. The broader compiled state was also evaluated against all 50,304 vocabulary logits. These are distinct checks: 518 explicit reader contractions and a full-vocabulary causal-effect comparison.

**Shared structure:** four binary splitting levels organized all 50,304 unembedding rows into 16 groups using only their weight directions. No activations, next-token labels or evaluation outcomes chose the groups. For each token,

    U_t = group_mean_t + token_residual_t.

The hierarchy is a root mean plus successive parent-to-child mean differences. This is an exact representation when the token residual remains. Omitting that residual is a substantive hypothesis, and it failed here. We compared shared group readers with a single root reader and shuffled memberships preserving group sizes; shuffled-group means were norm-matched.

Some nearest-centroid tokens look structured: one group includes “the/a/in/this,” another includes “74/68/77,” another “deploying/handing/showcasing,” and another “fantastic/innovative/remarkable.” But the leaves are very uneven: the largest contains 35,758 rows, about 71% of the vocabulary. These observations are descriptive, not semantic labels or evidence that all possible unembedding hierarchies fail. The null applies to this fixed 16-group shared-mean predictor.

## How the backward fold works

Write the MLP16 product features as z16=(L16 n16)*(R16 n16). Their output is D16 z16+b16. The raw input of MLP17 can be written

    r17 = a + lambda D16 z16,

where a includes the native residual background, MLP16 bias, embedding re-entry and attention17 output. With that background fixed, the next products are computed by

    left17  = L17 a + lambda (L17 D16) z16,
    right17 = R17 a + lambda (R17 D16) z16,
    z17 = left17 * right17 / (mean(r17²)+epsilon).

The token numerator is then

    U_t h17 = U_t a + lambda (U_t D16) z16
              + (U_t D17) z17 + U_t b17.

A group reader obeys the same formula. This folds across the adjacent bilinear modules without ignoring the denominator. The final residual h17 still undergoes RMS normalization and the model’s 30*tanh softcap. Token/group/residual pieces add before that shared nonlinear readout; capped logits do not generally add. All required native weights and background remain explicit.

The basic quadratic folding was already known in the module dossiers. This run tests its two-layer causal prediction and a weight-derived shared-reader hypothesis. It is not a new discovery that bilinear maps can be contracted.

## The causal test and results

For each of 48 previously opened sentence pairs, we swapped the donor’s complete MLP16 output into the recipient at its final semantic position. One run kept attention17 live; another held its output at that position to the native recipient value. The compiled fold predicts the latter exactly and is tested against the former.

| Panel | Individual-program live effect error | 95% paired-bootstrap interval | Group-only live effect error | CE-change prediction error |
|---|---:|---:|---:|---:|
| both/neither, bare frame | 4.56% | 3.80–5.60% | 94.37% | .00335 nats |
| both/neither, report frame | 5.61% | 4.73–6.55% | 96.18% | .00216 nats |
| either/not control task | 4.61% | 4.14–5.05% | 97.64% | .00256 nats |

Effect error is the Euclidean norm of candidate-minus-live change, divided by the native live change norm, after removing each vocabulary vector’s mean. CE error is mean absolute error in per-row correct-token cross-entropy changes. The intervals use 4,000 resamples of the 16 authored groups in each panel; they are not population guarantees.

All native endpoints were correct. The attention-fixed full-logit bridge has maximum absolute error at most 2.48e-5, with relative error below 7.4e-7. The token-plus-group-residual identity also passes. The registered instrument, native capability, live prediction and CE-prediction tests pass; shared-hierarchy sufficiency fails. The root and shuffled-group errors are approximately 100%, so the real groups improve a little but explain nowhere near enough.

The native MLP16 swap’s signed task recoveries are −.0216, −.0293 and −.0347. Those small negative effects matter to interpretation: we accurately predict an opposing contribution, not a component that recreates the target behavior. Selective removal, broader OOD transfer and independent extraction remain unproven.

The managed job ran at 14:58:58–14:59:01 UTC, making 12 body forwards over 192 sequence instances; recorded executor time was 1.34 seconds. Its temporary folded maps contain 47,545,895 additional coefficients. All 545,902,902 native parameters remain. This is algebraic access to readers, not a parameter saving.

## What the result changes

Folding farther back is operationally viable on these interventions: the intervening attention response contributes only a small error here. That gives us an explicit way to study individual downstream readers through an earlier product layer. However, coarse shared means throw away most vocabulary-specific response. Future structured-reader work must explain or retain that remainder, and distinguish operationally shared computations from nearby unembedding vectors. Increasing cluster count until the same test passes would not establish a new computation.

The [MLP17 dossier](../MLP17_CURRENT_UNDERSTANDING.md) now includes the older calibration, quadratic, context-gate and channel results as well as this outcome. The [MLP16 dossier](../MLP16_CURRENT_UNDERSTANDING.md) records its role and scope. The separate [OV pullback](attention_ov_input_reader_overlap.md) shows why an orthogonal head-output split can still share value readers.

Evidence: [registered mathematics and test](../../UNEMBEDDING_BACKWARD_VIEWS_V1_PREREGISTRATION.md), [native result](../../UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json), [paired audit](../../UNEMBEDDING_BACKWARD_VIEWS_AUDIT_V1_RESULT.json), [cluster representatives](../../UNEMBEDDING_BACKWARD_VIEWS_V1_CLUSTER_TOKENS.json), [hierarchy artifact](../../UNEMBEDDING_BACKWARD_VIEWS_V1_HIERARCHY.pt), [fold implementation](../../unembedding_backward_views_v1.py).
