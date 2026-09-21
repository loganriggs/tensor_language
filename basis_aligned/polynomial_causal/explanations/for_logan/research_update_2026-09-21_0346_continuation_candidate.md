# A weight-derived continuation candidate passes a new-document behavioral test

21 September 2026, 03:46 UTC.

**One of the stable extracted products now has a concrete behavioral hypothesis with fresh confirmation:** it supports non-whitespace alphabetic token continuations. Removing it hurts those predictions much more than spaced-word predictions. A norm-matched output-direction control produces a weaker effect, and differences remain when comparing contexts with the same current token.

This is a stronger candidate than “a compact feature with a consistent removal effect.” It is still short of an identified, reusable circuit: upstream extraction, direct original-operator grounding, broad OOD prediction and composition remain open.

## What the computation is

The candidate is group2 from the previous stable-group test. It reads two scalar directions from the normalized midpoint and preceding-MLP source inputs:

$$
u=a^\top n-\alpha,\qquad v=b^\top m-\beta,\qquad \widehat y=wuv.
$$

The vectors $a,b$ are readers; $w$ is the output writer; $\alpha,\beta$ are frozen centering constants. The upstream native model supplies $n,m$. The local program has one variable product and3,456 reader/writer coefficients, plus two constants and a shared output-coordinate mapping. That count does not include the upstream model.

This term came from a joint refactor of a weight-derived folded approximation, not from fitting the continuation labels. We subsequently inspected its behavior and froze the continuation hypothesis before confirmation.

## Discovery and a tokenizer correction

On the reused16-document discovery panel, group2 removal increased average next-token loss by0.01644 nats/token. Its strongest effects often followed word fragments or incomplete Unicode pieces. The paired stable group and another individual group had near-zero aggregate CE effects; stability alone did not imply comparable predictive importance.

The top256 vocabulary coordinates contain only9.81% of group2's centered writer energy. The registered output-localization prediction failed. Its meaning cannot be inferred from a short top-token list.

An initial strict byte decoder failed because one cached row started midway through a UTF-8 character. We corrected the annotation to resynchronize at that row boundary, validated the remaining complete byte stream strictly, and kept every model input and outcome unchanged. Decoding individual tokens as “�” would otherwise conflate incomplete bytes with punctuation.

The byte-aware discovery analysis found0.17943 continuation CE damage and essentially zero damage when the next token started with whitespace. However, a leave-document-out current-token lookup explained much of the scalar variation, making token identity an essential control.

## Frozen confirmation on unused documents

Before evaluation, we froze the program, token panel and thresholds. The panel contains32 new FineWeb rows, indices112–143 of the skip11000 cache, evaluated at positions16–255 in256-token contexts.

The registered conditions were precise:

- The prefix ends in an alphabetic character and has no unfinished UTF-8 sequence.
- **Continuation:** the next token starts with an ASCII letter, without whitespace.
- **Spaced word:** the next token starts with whitespace followed by an ASCII letter.

These are labels on observed next tokens, not input features given to the circuit. They support an operational token-continuation claim, not a complete linguistic definition of a word boundary.

| New-panel measurement | Real group2 removal | Norm-matched control |
|---|---:|---:|
| CE added at514 continuation sites | **0.12555** | 0.05249 |
| CE added at4787 spaced-word sites | **−0.00021** | 0.00179 |
| Continuation logodds decrease at continuation sites | **2.31797** | 1.16971 |
| Continuation logodds decrease at spaced-word sites | 0.08560 | 0.06286 |

CE is next-token cross-entropy, in nats/token; a positive increase is damage. Continuation logodds means log probability mass of vocabulary tokens beginning with an ASCII letter minus log probability mass of tokens beginning with whitespace and an ASCII letter.

The control uses the **same scalar activation** with group3's writer, scaled to match vocabulary-centered writer norm. It controls one aspect of perturbation magnitude and direction; one control direction cannot establish specificity against every possible perturbation.

All registered checks passed: instrument/count checks, continuation CE damage above0.05 with spaced-word damage below0.02 in magnitude, and a real-versus-control logodds difference above0.02. Programs were removed from the original native final residual with final normalization and softcap intact.

## Does this go beyond current-token identity?

The secondary test compares continuation and spaced-word contexts within the same current-token ID. It covers34 token types,51 continuation sites and257 spaced-word sites. With weights equal to the smaller class count within each token stratum, the mean difference in real removal-induced logodds decrease is **1.59844**.

A paired32-document bootstrap gives a descriptive95% interval of **0.858–2.021**. Thus the effect is not explained solely by which token immediately precedes the prediction. This matching does not control every difference in context or prediction difficulty, and it is not a randomized manipulation of a continuation condition.

Other descriptive intervals from10,000 document draws:

| Quantity | 95% interval |
|---|---:|
| Real continuation CE damage | 0.07545–0.17177 |
| Real spaced-word CE damage | −0.00134–0.00090 |
| Real minus control continuation CE damage | 0.04322–0.10517 |
| Real minus control continuation logodds decrease | 1.02075–1.27977 |

These are conditional panel intervals, not simultaneous or population guarantees.

## What remains before calling it a circuit

The candidate has an executable computational specification, stable removal effects across two fitted definitions, and a frozen behavioral prediction confirmed on unused documents. That is meaningful progress.

However, the decomposition has several approximation stages. A legal removal of this extracted term is not by itself proof that it is a unique summand of the original native computation. The next priority is to compare it directly with an original-weight observable and trace its scalar inputs backward. Broader OOD tests, controlled examples, collateral-behavior checks and joint composition are also still required. We have not established monosemanticity.

See the [candidate dossier](../../direct_tensor_match/circuits/letter_continuation_group2.md) for the exact boundary, evidence ledger and next checks.

Evidence: [discovery profile](../../direct_tensor_match/MIDPOINT_STABLE_GROUP_PROFILE_V1.json), [byte-aware audit](../../direct_tensor_match/MIDPOINT_STABLE_GROUP_BOUNDARY_AUDIT_V1.json), [frozen confirmation plan](../../direct_tensor_match/MIDPOINT_CONTINUATION_GROUP_PLAN_V1.json), [native results](../../direct_tensor_match/MIDPOINT_CONTINUATION_GROUP_NATIVE_V1.json), and [uncertainty audit](../../direct_tensor_match/MIDPOINT_CONTINUATION_GROUP_AUDIT_V1.json). The managed32-capture confirmation run finished successfully in approximately2.9 seconds. FineWeb freshness is relative to this decomposition study; pretraining overlap is unknown.
