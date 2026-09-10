# What the context test changes about our proposed circuit

The weight-based decomposition is executable, but we have not extracted a
context-independent grammatical computation. The latest test gives two reasons
to keep that distinction clear: the model often answers these longer sentences
incorrectly, and the saved circuit coordinates combine cue and context through
an interaction that simple addition cannot express.

The high-level sequence was:

1. The earlier saved circuit transferred well on new word combinations in short
   sentences. Folding its learned directions into the actual attention weights
   gave an exact way to read and edit its14 scalar coordinates.
2. Separating its first-layer and current-layer value sources did not produce
   sufficient circuits. Both sources and their downstream interaction mattered.
3. Before testing a completed inner correlative, I found that v493/v495/v497 had
   already done that experiment. Their comparison phrase differed in length and
   wording. I corrected the proposed next step and tested one matched word change.
4. This test failed its native capability gate. A CPU audit also found that the
   older report's “32/32 kept” statistic measured a positive cue effect, not
   correct answers at both ends. Its claim that the model tracks the still-open
   correlative was stronger than those saved statistics support.
5. A separate mathematical audit rejected additive cue/context separation of the
   existing coordinates. This is a useful constraint on an explanation, not a
   newly discovered parser or a completed decomposition.

The governing direction remains the original bilinear handoff and pilot, with
the appended requirement to discover a reusable computation and account for
its inputs, consumers and remaining opaque weights. The is/was direction was
not reopened for this work.

## What was compared

For16 reporter groups in each of two sentence frames, the four combinations were:

    The scientist praised both the crate holding only the rope or the lamp
    The scientist praised neither the crate holding only the rope or the lamp
    The scientist praised both the crate holding either the rope or the lamp
    The scientist praised neither the crate holding either the rope or the lamp

The second frame starts “In the notes the scientist named ...”. The intended
next token is “and” after the outer “both” and “nor” after “neither”. Replacing
“either” with “only” changes exactly one token. Token count, cue position, final
token and all other words are identical. This removes a length-only explanation
for this contrast; it does not separate all lexical and syntactic explanations.

The circuit was frozen:26 selected attention heads, grouped into14 blocks with
one learned scalar direction per block. No new fit, head, rank, gain or mixing
coefficient was chosen. All native upstream computation remained active.

The managed GPU run took24 body forwards covering384 sequences. The recorded
executor time was1.616 seconds; runner timestamps were13:56:06–13:56:10 UTC.
Native versus instrumented baseline logits agreed exactly. The folded scalar
checks passed their registered numerical tolerances.

## The model's own answer is the first limitation

A pair is correct only if the native model prefers the intended answer in both
the “both” and the “neither” sentence. The new counts were:

| Frame | Inner word | Pairs correct at both endpoints |
|---|---|---:|
| “The scientist praised ...” | only | 0/16 |
| “The scientist praised ...” | either | 0/16 |
| “In the notes ... named ...” | only | 3/16 |
| “In the notes ... named ...” | either | 8/16 |

The model prefers “and” on both cue values throughout the first frame. For
example, its “and minus nor” margin is about7.49 after “both” and2.14 after
“neither” in the first discharged example. The cue changes the score, but it
does not change the answer to the intended “nor”.

This explains the old metric problem. Suppose the model always prefers “and”,
by75 score units after “both” and22 after “neither”. Across the two reciprocal
swap directions, the old donor-oriented average axes are−26.5 and+26.5, and
every recovery denominator is positive. Yet zero pairs have two correct
answers. The integer example is an exact CPU-checked counterexample.

The older v493 aggregate cannot recover the original32-row endpoint accuracy.
Its positive-effect counts and averaged axes alone do not establish the
reported grammatical capability. The new same-template examples demonstrate
the limitation directly. A shared endpoint-summary helper now reports effect
denominators and actual endpoint correctness separately.

## What the interventions showed

A donor swap replaces the recipient's circuit coordinate with the corresponding
coordinate measured in the other sentence, then recomputes the native suffix.
Raw recovery measures the fraction of the native cue-induced margin change
reproduced by this intervention. The historical normalized recovery divides
that number by the recovery from swapping all26 selected heads.

| Frame | Inner word | Full-head raw recovery | Scalar normalized recovery |
|---|---|---:|---:|
| Praise | only | .931 | .761 |
| Praise | either | .884 | .681 |
| Notes | only | .928 | .753 |
| Notes | either | .895 | .753 |

The registered neutral threshold was.8 in both frames. Both miss it. The
registered extra deficit from the completed inner cue was.15 in each frame;
the measured deficits were.080 and−.0007. Paired bootstrap intervals over the
16 authored groups were[.0736,.0875] and[−.0073,.0066]. These intervals describe
this finite panel, not model-training or population generalization.

Same-answer either/only swaps caused small margin movements, .039–.063 of the
recipient's median native cue separation. Their registered invariance predicate
still fails because it requires native capability. This is inconclusive
semantic evidence, not measured large collateral damage. The overall result
is a valid instrument with a failed behavioral hypothesis.

## The mathematical result: cue and context do not simply add

Let s(c,k) be one saved block scalar, with outer cue c and inner word k. A simple
reusable decomposition would say

    s(c,k) = cue_part(c) + context_part(k).

That predicts the cue difference is identical in both contexts. Define the
mixed difference:

    D = s(neither,either) - s(both,either)
        - s(neither,only) + s(both,only).

Any additive description has D=0. Conversely, on exactly these four corners,
D=0 is sufficient for an additive table. If D is nonzero, no choice of the two
parts can remove it. The best additive approximation has total squared error
D²/4 across the four scalar values; at least one corner must err by |D|/4.
This follows by projecting the table onto the vector(1,−1,−1,1), which is
orthogonal to every additive table. An independent least-squares computation
verified the formula to1.34e−15 on planted numerical examples.

All448 tested layer/reporter contrasts—14 layers ×16 groups ×2 frames—exceeded
the sum of their four measured native-versus-folded scalar discrepancies.
Thus numerical disagreement in the folding bridge does not explain them.
When weighted by each block's actual output writer, the mixed-difference norm
was.416 and.281 of the corresponding cue-difference norm in the two frames.
These are norms of writes stacked across blocks, not final-logit errors.

This rejects additive separation of the **fixed coordinates** on these inputs.
It does not reject nonlinear encodings, a context-dependent reader, or a
different circuit. Small context-swap effects can coexist with these interactions
because the native suffix need not respond equally to every scalar change.

## What this contributes to the four desired properties

The short-frame new-combination transfer and exact intervention compiler remain
supported. The current extension adds a negative generalization result and a
precise constraint on the producer we would need to extract. It provides no
new independent extraction, selective removal or shared semantic reuse.

All545902902 native parameters remain necessary, alongside the existing76032
folded reader/writer coefficients. There is no structural saving. The productive
math question is now which explicitly describable context-dependent operation
produces these coordinates; a fit with another rank or gain would not answer it.
Before a stronger circuit claim, endpoint capability and the proposed operation's
intervention semantics both need evidence.

Primary receipts: `CORRELATIVE_MATCHED_CONTEXT_V1_RESULT.json`,
`CORRELATIVE_MATCHED_CONTEXT_AUDIT_V1_RESULT.json`, and
`CORRELATIVE_ENDPOINT_CAPABILITY_AUDIT_V1_RESULT.json` in the parent directory.
The literature mapping and full derivation are in
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1349.md`.
