# Noun-number selection: what the model does, and what the math rules out

The new tests move beyond the stagnant is/was investigation. We first specified a simple computation—choose the noun whose number controls a reflexive—and checked whether the model actually performs it. It does not reliably switch the controlling noun with the verb. In short two-noun sentences, all 128 measured preferences follow the second noun. Adding a third noun then breaks each of four simple rules we had registered in advance.

The useful mathematical lead is **context-dependent combination of noun-number signals**. Making the third noun human reduces the second noun's influence and increases the third noun's influence. We have verified that the former interaction cannot originate in a token-local lookup. We also constructed a counterexample showing why it could still be created by final normalization, rather than an internal bilinear routing operation. No new circuit satisfying all four requested properties has been identified.

## Relation to the original handoff and pilot

The controlling sources are the [original handoff](bilinear_circuit_reconstruction_codex_handoff.md) and [pilot report](bilinear_reconstruction_pilot_report.md). Their updated criterion is to recover a previously unspecified reusable computation with explicit inputs and consumers, verify extraction and joint interventions on held-out cases, and reduce structural description cost while charging adapters and opaque weights.

The pilot established a faithful way to execute bilinear attention. It did not discover a smaller semantic algorithm. Likewise, folding a bilinear MLP into a downstream linear reader is mathematically available: if

\[
f(u)=D[(Lu)\odot(Ru)]+b,
\]

then a linear consumer C can use CD and Cb directly. Here u is the normalized input vector; L and R produce the two factors; their elementwise product is written through D. This rewrite is exact over that specified interface. Intervening normalization or contextual attention must remain explicit; a matrix fold cannot erase them.

For two consumers C1 and C2, their pulled-back products identify candidate shared and private computations. Shared use of a native head or MLP is insufficient: we need to show that the same produced variable serves both consumers, and that their independent and joint edits have predictable effects. The present tests supply a sharper candidate operation before another weight decomposition is attempted.

## 1. A controller selector with an explicit formula

Consider these sentence prefixes:

- “The king promised the kings to defend …”
- “The king persuaded the kings to defend …”

The intended reflexive is *himself* for the first and *themselves* for the second: promise uses the subject as the understood actor; persuade uses the object. This control-verb distinction is described in [Demestre's primary study](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1320966/full).

Let s and o denote subject and object number, each −1 for singular and +1 for plural. Let c=+1 for promised and −1 for persuaded. The intended selector is

\[
y(c,s,o)=\frac{s+o+c(s-o)}2.
\]

When c=+1 this is s; when c=−1 it is o. This is a bilinear task specification: the verb must change how number information is used. Its coefficient matrix on (1,c) and (s,o) is

\[
\frac12\begin{pmatrix}1&1\\1&-1\end{pmatrix},
\]

with determinant −1/2 and rank two. This is the rank of the proposed task formula, **not a measurement of the trained model's tensor rank**.

We evaluated all eight combinations of c,s,o in 16 noun/action worlds: 128 prefixes total. Within each world, the two noun slots use the same noun stem. Reversing which noun is plural preserves the bag of token identities, so token counts alone cannot distinguish those opposite-number role assignments. All prefixes in a world have equal token length.

We measured m=logit(themselves)−logit(himself), after the native output transformation. This is preference between two answers, not full-vocabulary top-1 accuracy. Each registered corner cell had to attain at least 75% correct signs and mean signed margin at least 1. No examples were filtered.

**Result:** the grammatical-controller hypothesis failed. The separately registered second/nearest-noun rule passed, with 128/128 signs following object number. All 32 promised cases with opposite noun numbers disagreed with the intended controller. This does not establish a general nearest-human rule: object, second noun, nearest noun, nearest human, and fixed position coincide on these prefixes.

## 2. What the finite-cube coefficients mean

For a fixed lexical world, eight observations give the exact expansion

\[
m=a_0+a_c c+a_s s+a_o o+a_{cs}cs+a_{co}co+a_{so}so+a_{cso}cso.
\]

Each coefficient is the average of the observed margin times its named product over the eight corners. This is the Walsh expansion, an orthogonal coordinate system for a finite table. It is exact on that table; it does not prove that the full transformer is this polynomial on arbitrary inputs.

The held-design worlds had mean subject coefficient 0.739 and object coefficient 2.526, versus verb×subject 0.011 and verb×object −0.144. The object signal dominates, and verb-dependent switching is weak. Dropping all interaction terms preserves all 128 object-number signs. Interaction root-mean-square magnitude was only 6.2–15.8% of total nonconstant magnitude across worlds. Parseval's identity—squared table error equals the sum of squared omitted coefficients—was checked numerically.

This is a descriptive projection using every opened corner, not a fitted predictor tested on unseen values. The evidence changes the proposed operation from grammatical selection to the model's actual combination of number signals.

## 3. The third noun distinguishes the simple alternatives

We added a noun in a prepositional phrase:

> The king persuaded the kings near the brother to defend …

Subject number, object number, third-noun number, verb, and whether the third noun is human or inanimate were varied independently: 32 corners in each of eight worlds, totaling 256 new prefixes. The four prospectively specified rules were grammatical controller, object number, nearest noun, and nearest human noun.

**All four failed their registered cellwise gates**, including the diagnostic groups restricted to either animacy category. Pooled sign counts were 173/256, 211/256, 173/256, and 219/256 respectively. The largest count does not promote the nearest-human rule; pooling would hide its failing conditions. We did not change the phrases, thresholds, or rule definitions to rescue a result.

Let a be the third noun's number and h=+1 for human, −1 for inanimate. The five-factor expansion includes these mean coefficients:

| Term | Mean coefficient | Meaning within the opened table |
|---|---:|---|
| o | 1.367 | Second-noun number influence |
| a | 0.840 | Third-noun number influence |
| oh | −0.313 | Human category reduces second-noun number influence |
| ah | 0.287 | Human category increases third-noun number influence |

These are averages over the other independently varied factors; higher interactions also exist. The root-mean-square oh coefficient is 23.19% of the root-mean-square o coefficient. This is a material contextual modulation, not a hard selector or a named neural mechanism.

## 4. An exact test for lexical versus cross-token interaction

For two binary factors, define the mixed difference

\[
\Delta_{oh}f=f_{++}-f_{+-}-f_{-+}+f_{--}.
\]

At any token position, a lookup map has zero mixed difference for every possible choice of embedding weights whenever the two diagonal token multisets agree. We checked this directly from the frozen token IDs.

Object number changes token position 4; human/inanimate category changes position 7 (zero-based indices). All **64 oh squares** have matching diagonal token multisets at every position. Therefore embedding lookup, per-token embedding normalization, and linear combinations of those token-local vectors cannot create this mixed term by themselves.

In contrast, third-noun number and category both change position 7. None of the **64 ah squares** satisfies universal token-local cancellation. Four different words can have an interaction already in their embedding lookup. Calling ah evidence of a learned multiplication would be unjustified.

Nonzero output oh still does not prove internal attention gating. As an executable counterexample, take the additive two-coordinate state

\[
x(o,h)=(1+o/4,\ 1+h/3).
\]

Its mixed coefficient is zero. Reading its first coordinate after root-mean-square normalization produces mixed coefficient **0.0207664**. A nonlinear decoder can create interaction even when the incoming state combines factors additively.

The next informative native measurement is therefore where the cross-token interaction forms: compare residual states, normalized states, and decoded outputs before attributing it to attention or a bilinear MLP. This measurement is not yet registered or queued. Only a material, localized internal operation would justify producer/consumer weight folding and subsequent extraction, removal, and shared-use tests.

## Evidence, cost, and current limits

Both native screens passed their instrument checks. Controller: 8 forwards / 128 sequences, 0.505 seconds executor time. Third noun: 16 forwards / 256 sequences, 0.558 seconds. These timings exclude research, preflight, and queue time. There were no fits or backward passes. The model retained all 545,902,902 parameters: structural savings remain zero.

The subsequent CPU audit reused the existing Walsh transform with explicit factor-sign/bit ordering. It reproduced both coefficient tables within 1.78e−15, reproduced every third-noun rule verdict, verified all token-support certificates, and executed the normalization counterexample. No further native forwards were used.

- [Controller result](../SUBJECT_OBJECT_CONTROLLER_V1_RESULT.json) and [additivity audit](../CONTROLLER_ADDITIVITY_AUDIT_V1_RESULT.json).
- [Third-noun protocol](../THIRD_NOUN_ANIMACY_V1_PREREGISTRATION.md) and [result](../THIRD_NOUN_ANIMACY_V1_RESULT.json).
- [Shared factorial helper](../factorial_semantic_support_v1.py), [executed audit](../audit_factorial_semantic_support_v1.py), and [audit receipt](../FACTORIAL_SEMANTIC_SUPPORT_V1_RESULT.json).
- [Hourly review](../HOURLY_STRATEGIC_REVIEW_2026-09-10_0514.md): median valid-receipt interval exceeded the ten-minute target; this shared replay/support tool is the required bounded workflow repair.

OOD behavior was tested for the simple rules and failed. Independent circuit extraction, selective removal, and reusable composition remain unestablished. The progress is a better specified candidate computation and a mathematical test that prevents mistaking lexical or decoder interactions for the desired internal circuit.
