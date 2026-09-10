# Cue/context composition still needs the upstream computation

The trained-model four-corner test is now complete and valid. MLP4's local
products help predict a cue change in a different context, but they do not
explain its context-dependent correction: that prediction still has60–85%
relative full-output error. No circuit has been promoted.

The next useful mathematical result is more specific than another failed
approximation. For these matched texts, the direct token-embedding contribution
to the mixed cue/context input is exactly zero. Any remaining mixed raw input
must arrive through contextual attention/MLP writes. Normalization can then
create additional interaction. A tested source-capture tool now lets us trace
those contributions before MLP4.

## What the native experiment asked

We used35evaluation squares, each containing four texts: two cue values crossed
with two lexical backgrounds. For example, an is/was cue can change this to
that while a separate token position changes the lexical context. The context
can contain several changed words; it is not necessarily one subject feature.
All texts were already in the opened datasets.

For every square, we tried predicting each corner from the other three. The
candidate transferred the cue's MLP4 output change from the other context and
added the exact bilinear product between the separately observed cue and
context changes. Every prediction was inserted at MLP4 and followed by full
native downstream computation. No attention head or value port was frozen.

We tested two different requirements at the unchanged10% preliminary threshold:

- Predict the complete source cue-swap effect.
- Predict the correction to simply transferring the cue change between contexts.

Both comparisons used the entire centered50,304-vocabulary logit vector and
the answer-minus-foil margin. The second requirement prevents a nearly
context-independent cue effect from passing as an explanation of interaction.

## What passed and failed

| Evaluation panel | Local prediction: total effect error | Local prediction: interaction-correction error |
|---|---:|---:|
| has/had A2 | 43.2–47.1% | 64.8–73.1% |
| has/had held-out | 49.0–62.0% | 74.7–83.9% |
| is/was A2 | 5.4–6.4% | 62.0–70.4% |
| is/was held-out | 7.3–15.0% | 59.8–85.2% |

These are relative full-vector errors across the four missing-corner directions
in each panel. The full-vector and margin requirements together pass for the
total effect in5of16cells, but for the interaction correction in0of16.

The is/was A2 panel illustrates why the distinction matters. Even plain cue
transfer without the bilinear correction has only8.5–9.0% full-vector error
there. Improving total prediction therefore does not demonstrate that we
understand the smaller context-dependent component. The other panels also
prevent promoting this as a generally portable cue computation. Differences
between task panels should not be read as an intrinsic complexity comparison:
the lexical-background changes differ.

The inherited-input-only diagnostic also fails to reproduce the correction.
Both local products and nonadditive inputs contribute. That observation does
not identify a reusable semantic operation or establish which earlier layer
constructed the incoming interaction.

## Why there are two result files

V1 remains marked INVALID. Its FP64 weight formulas differed from deployed
FP32 MLP outputs by up to.004272 in absolute value, although the relative
module discrepancy was only about2e-7. The strict absolute local checks failed.
Two small-effect margin checks also narrowly failed when compared across
readout batch shapes. Final-logit equivalence checks were already within
2.15e-5, but that did not license ignoring the failed instruments.

V2 corrected the numerical comparisons without changing the scientific
candidate or thresholds. It replayed the actual FP32 weight formula exactly,
kept explicit native-minus-FP64 roundoff in the algebraic oracles, and replayed
the old positive control using its original batches and semantic-only readout.
The fourth-corner roundoff enters only the already fourth-dependent exact-sum
control. It never enters the three-corner candidate. The independent candidate
oracle uses only roundoffs from the three known corners.

The deployed weight replay, exact-sum logit replay, independent-candidate logit
replay and original parent margin replay are all exactly equal in V2. The
unchanged V1 candidate margin effects also replay with maximum drift0. The
full-logit comparison between the two readout geometries differs by at most
3.44e-5 and passes the original numerical bars. The FP64 partition identity
has maximum error2.30e-11 and passes its1e-9bars.

The valid run used144forwards on1,408sequence instances plus16local MLP
recomputations, in3.54seconds of executor time. Including the preserved
invalid attempt, the two runs used272forwards on2,528sequence instances.
No weights were trained or replaced permanently. All545,902,902native
parameters remain charged; actual structural saving is zero.

## The new math: where can the mixed input originate?

The model repeatedly scales the residual stream, injects the original token
embedding, and adds attention and MLP outputs. Consequently the raw input x
to MLP4, before its input normalization, can be written as

    x = alpha * e
        + sum_(l=0..4) gamma_attention_l * attention_write_l
        + sum_(l=0..3) gamma_mlp_l * mlp_write_l.

There are ten terms. The scalar coefficients follow directly from the learned
residual multipliers. In the current checkpoint, alpha is18.11378644; the
coefficients for attention/MLP writes0,1,2,3 are respectively
.00666975,.52537125,.26580048,.462890625, and attention4 has coefficient1.
These transport coefficients are not causal importance scores.

Define a mixed difference as the cue change in context1 minus the cue change
in context0:

    Delta_square x = (x11-x10) - (x01-x00).

Because the residual sum is linear in its writes, its mixed difference is the
same weighted sum of their mixed differences. The contextual writes themselves
can be highly nonlinear functions of the text; the identity does not discard
the computations that produced them.

For the direct embedding term there is a stronger statement. At every token
position in these squares, the token changes with either the cue or the
context, never both. The two diagonal token multisets therefore agree:

    {token00, token11} = {token01, token10}.

Any function of that single token has zero mixed difference. This includes
the learned token lookup and its per-token RMS normalization. Therefore

    Delta_square e = 0,

and the explicit embedding term drops out of this mixed raw-input equation.
This was certified directly from token IDs at all326valid positions in the
46fitting/evaluation squares, with no trained-model forward pass. It is not a
finite-sample activation-rank claim. It applies to these disjoint-edit squares;
an overlapping-edit counterexample in the controls has nonzero embedding mix.

The zero does not remove embeddings from the program. Attention and MLP writes
still depend on them, and the normalized MLP4 input is a nonlinear function
of the complete raw input. A zero mixed embedding contribution is narrower
than embedding ablation or independent extraction.

## What is ready for the next native test

The new capture tool follows the original backend's residual operation order
and records the ten raw sources before MLP4. On a tiny model, the recomputed
normalized input is bitwise equal to the native input; the source sum and its
mixed difference close within4.45e-16. Contextual mixed input remains live while
the direct embedding contribution is exactly zero.

The cancellation check initially exposed floating-point association: sequential
four-term subtraction can leave tiny residue even when the real-arithmetic
identity is zero. Subtracting the two equal cue edges first makes the expected
zero bitwise. The final control keeps a separate live mixed-input example, so
exact embedding cancellation cannot make the whole instrument vacuous.

The next scientific measurement is to trace the inherited signal through these
contextual writes, particularly the incoming residual versus the new attention4
write. Such edge-localization measurements must retain the distinction between
editing a captured contribution and independently removing its producer.
They are not a renormalized rescue of the failed three-corner predictor. No
native source-localization protocol or successor job is registered yet; the
CPU capture checks and token certificate are the concrete continuation already
completed. The full circuit goal remains active.

## Evidence

- [Valid native result](../../BILIN18_MLP4_SEMANTIC_SQUARE_V2_RESULT.json)
- [Preserved invalid attempt](../../BILIN18_MLP4_SEMANTIC_SQUARE_V1_RESULT.json)
- [Original protocol](../../BILIN18_MLP4_SEMANTIC_SQUARE_V1_PREREGISTRATION.md)
- [Numerical correction, with unchanged scientific criteria](../../BILIN18_MLP4_SEMANTIC_SQUARE_V2_NUMERICAL_CORRECTION.md)
- [Residual lineage capture](../../four_corner_residual_lineage.py)
- [Executed lineage controls](../../FOUR_CORNER_RESIDUAL_LINEAGE_V1_CONTROLS.json)
- [Native coefficients and token-level certificate](../../FOUR_CORNER_EMBEDDING_LINEAGE_V1_AUDIT.json)
