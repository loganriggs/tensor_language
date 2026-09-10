# Noun-number selection: what the model does, and what the math rules out

The new tests move beyond the stagnant is/was investigation. We first specified a simple computation—choose the noun whose number controls a reflexive—and checked whether the model actually performs it. It does not reliably switch the controlling noun with the verb. In short two-noun sentences, all 128 measured preferences follow the second noun. Adding a third noun then breaks each of four simple rules we had registered in advance.

The useful mathematical lead is **context-dependent combination of noun-number signals**. Making the third noun human reduces the second noun's influence and increases the third noun's influence. We have verified that the former interaction cannot originate in a token-local lookup. A subsequent native intervention now shows that most of the answer interaction is carried in the internal residual state: removing it leaves only 7.1–11.1% of its original magnitude. Raw-vector accounting favored MLP writes, but the subsequent behavioral test rejected them as the main answer carrier: removing all their mixed writes leaves 68–110% of the answer interaction. The subsequent attention test also fails sufficiency alone, leaving 27–43%, while attention and MLP removals compose almost additively. Reader-weighted localization identifies layer 9 attention as the largest individual contributor in every world, primarily through the answer logits rather than normalization. This still does not identify a reusable operation or a circuit satisfying all four requested properties.

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

The next informative native measurement is therefore where the cross-token interaction forms: compare residual states, normalized states, and decoded outputs before attributing it to attention or a bilinear MLP. That measurement has now been registered and executed; its result is below. Only an identified producer/consumer operation would justify claiming extraction, selective removal, and reusable composition.

## Evidence, cost, and current limits

Both native screens passed their instrument checks. Controller: 8 forwards / 128 sequences, 0.505 seconds executor time. Third noun: 16 forwards / 256 sequences, 0.558 seconds. These timings exclude research, preflight, and queue time. There were no fits or backward passes. The model retained all 545,902,902 parameters: structural savings remain zero.

The subsequent CPU audit reused the existing Walsh transform with explicit factor-sign/bit ordering. It reproduced both coefficient tables within 1.78e−15, reproduced every third-noun rule verdict, verified all token-support certificates, and executed the normalization counterexample. No further native forwards were used.

- [Controller result](../SUBJECT_OBJECT_CONTROLLER_V1_RESULT.json) and [additivity audit](../CONTROLLER_ADDITIVITY_AUDIT_V1_RESULT.json).
- [Third-noun protocol](../THIRD_NOUN_ANIMACY_V1_PREREGISTRATION.md) and [result](../THIRD_NOUN_ANIMACY_V1_RESULT.json).
- [Shared factorial helper](../factorial_semantic_support_v1.py), [executed audit](../audit_factorial_semantic_support_v1.py), and [audit receipt](../FACTORIAL_SEMANTIC_SUPPORT_V1_RESULT.json).
- [Hourly review](../HOURLY_STRATEGIC_REVIEW_2026-09-10_0514.md): median valid-receipt interval exceeded the ten-minute target; this shared replay/support tool is the required bounded workflow repair.

OOD behavior was tested for the simple rules and failed. Independent circuit extraction, selective removal, and reusable composition remain unestablished. The positive internal-state test below improves localization without promoting the failed behavioral selectors.


## 5. Native interaction removal passes: most of the answer effect is internal

We subsequently registered and executed [THIRD_NOUN_MIXED_STATE_V1](../THIRD_NOUN_MIXED_STATE_V1_PREREGISTRATION.md). For every fixed combination of verb, subject number and third-noun number, the projector

\[
P_{oh}x=(x-\operatorname{flip}_o x-\operatorname{flip}_h x+
\operatorname{flip}_{oh}x)/4
\]

isolates the part of the final residual that depends jointly on object number and human category. A flip exchanges the two factor settings while keeping the other settings fixed. This retains interactions with the fixed context too; it is more complete than removing only the mean oh coefficient.

We decoded both x and x−P_oh x using the unchanged final normalization, vocabulary weights and softcap. Across all eight worlds:

- Native mixed answer-margin root-mean-square magnitude was 0.243–0.465 logit units.
- Removing the internal mixed state left **7.08–11.11%** of that mixed margin magnitude, passing the predeclared 25% maximum in every world.
- The removed effect's signed projection onto the original mixed answer effect was **89.83–93.81%**.
- Full-vocabulary mixed-output magnitude remaining was 7.32–44.02%. The answer effect and the complete output vector are different observables.

The native parent replay was exact. Independently decoding the saved final state differed by at most 1.05e−5 logits, relative error 1.58e−6, within both registered tolerances. There were 16 native forwards over 256 prefixes plus 32 decoder batches over 512 saved/edited final states; executor time was 0.919 seconds. No fitting or backward passes occurred.

This rejects the idea that the answer interaction is created mainly by the final decoder acting on an interaction-free residual. It does **not** identify a neuron or extracted semantic variable. The removed component uses four observed states for each context. It is a synthetic, factor-specific state edit, with all native producers and decoder retained. Selectivity on unrelated behaviors remains untested.

## 6. Exact native source accounting points to MLP writes

These are vector-space measurements. The behavioral test in section 7 shows why they cannot select the answer-carrying circuit by themselves.

The native residual recurrence gives

\[
x_{18}=k_e e+\sum_{l=0}^{17}k_l(A_l+M_l),\qquad
k_l=\prod_{j=l+1}^{17}\lambda_{0,j}.
\]

Here e is the normalized token embedding, A_l and M_l are actual attention and MLP writes at the readout position, and each k_l is the product of later residual-carry coefficients. The embedding coefficient k_e also includes repeated embedding injection. This accounts for the writes actually made in the native run; their contextual input dependencies remain present.

Apply P_oh to each transported write to obtain vectors d_i. Save their Gram matrix G_ij=⟨d_i,d_j⟩, which records their lengths and mutual alignment. If t=Σ_i d_i is the final mixed state, a source's signed contribution is

\[
p_i=\frac{\langle d_i,t\rangle}{\|t\|^2}
=\frac{\sum_jG_{ij}}{\sum_{jk}G_{jk}}.
\]

These signed contributions sum to one; they can be negative because writes can cancel. The executed CPU audit found:

| Native write group | Signed projection onto final mixed state, across worlds |
|---|---:|
| All MLP writes | 85.35–95.22% |
| All attention writes | 4.78–14.65% |
| All writes in layers 12–17 | 80.53–91.64% |
| Last MLP alone | 40.99–64.28% |

The last MLP is the largest individual source in every world, but by itself its relative vector error is 49.70–71.39%. The sum of individual write lengths is 2.28–3.06 times the length of their sum, demonstrating substantial cancellation. Choosing the largest write is therefore not an extracted circuit.

The first attention write at this unchanged query satisfies the registered cross-source zero check, up to floating-point roundoff. The immediately following normalization/MLP produces a nonzero mixed state. This is consistent with attention transporting separate source information and a later token-local nonlinear operation combining it. It does not yet distinguish that normalization from the bilinear product, nor identify which input coordinates carry the two factors.

The next decomposition tested was **inherited interaction versus newly formed interaction at a producer/consumer interface**; section 7 gives its completed outcome. MLP dominance in direct-write accounting does not make upstream attention unnecessary. A future test must preserve those dependencies, distinguish normalization effects, and validate an explicit computation before claiming shared reusable factors. No successor GPU job is currently queued.

The source telescope's relative error was 8.56e−8 for the raw final state and 2.22e−6 for the mixed state. CPU Gram accounting reproduced squared mixed-state norms within 9.40e−7 relative error and passed a cancellation control with both positive and negative contributions. These are numerical identities and native attribution, not independent causal-source interventions.

Receipts: [native mixed-state result](../THIRD_NOUN_MIXED_STATE_V1_RESULT.json), [CPU source audit](../third_noun_source_gram_audit_v1.py), and [source audit result](../THIRD_NOUN_SOURCE_GRAM_AUDIT_V1_RESULT.json). All 545,902,902 native parameters remain charged, with zero structural savings. The CPU source analysis was claimed and executed after interpreting the native result.


## 7. Symmetric MLP factorization is exact, but fails the behavioral carrier test

We split each MLP's actual Left and Right activations into four components while holding the other three sentence factors fixed:

\[
L=L_0+L_o+L_h+L_{oh},\qquad R=R_0+R_o+R_h+R_{oh}.
\]

The subscripts indicate which factor flips change that component's sign. These components are computed symmetrically from all four corners. No sentence is privileged as the baseline. The joint component of their elementwise product is exactly

\[
P_{oh}(L\odot R)=
\underbrace{L_o\odot R_h+L_h\odot R_o}_{\text{formed at the product}}
+
\underbrace{L_0\odot R_{oh}+L_{oh}\odot R_0}_{\text{inherited at the input}}.
\]

Applying the native Down weights and residual-carry coefficients gives two sets of output writes across all 18 MLPs. “Inherited” includes interactions introduced by the MLP input's normalization; it does not establish an upstream semantic variable. “Formed at the product” names this algebraic interface, not a newly discovered algorithm.

The partition passed its instrument checks: exact factor algebra relative error at most 2.01e−17, and agreement with native FP32 mixed MLP writes within 1.55e−5 relative error. CPU controls verified pure new/inherited examples, factor rescaling, and row permutation. No neuron, layer, rank, or phrase was selected to improve the outcome.

We then removed each transported branch from the final native residual and decoded it with the unchanged model:

| Removed contribution | Remaining native answer-interaction RMS ratio |
|---|---:|
| All MLP mixed writes | **0.681–1.103** |
| Product-formed branch only | 0.765–1.030 |
| Input-inherited branch only | 0.693–1.191 |

Every branch failed the registered requirement of at most 0.25 in every world. In particular, the complete MLP branch fails despite accounting for 85–95% of the final mixed state's signed vector projection. This rejects using that vector projection as evidence that MLPs carry the main answer interaction. It does not refute the previous positive removal of the *whole* internal mixed state.

The run used 16 native forwards / 256 prefixes and 64 decoder batches / 1,024 states in 1.078 seconds. Parent replay was exact; the decoder bridge again passed both absolute and relative tolerances. No parameters were fit or removed from storage. [Protocol](../THIRD_NOUN_MLP_FACTOR_PARTITION_V1_PREREGISTRATION.md), [result](../THIRD_NOUN_MLP_FACTOR_PARTITION_V1_RESULT.json), and [factor algebra](../symmetric_factor_interaction_v1.py).

## 8. Replace vector-size ranking with exact reader-weighted edit accounting

For a final state x, source writes d_i and edit amounts alpha_i, define

\[
x'=x+\sum_i\alpha_i d_i.
\]

Let W contain the actual vocabulary-reader rows under study. Before softcap, the decoder depends on its reader numerators and a common normalization denominator:

\[
W x'=Wx+\sum_i\alpha_i Wd_i,
\]

\[
\|x'\|^2=\|x\|^2+2\sum_i\alpha_i\langle x,d_i\rangle
+\sum_{ij}\alpha_i\alpha_j\langle d_i,d_j\rangle.
\]

The native decoder applies root-mean-square normalization using sqrt(||x'||²/D+epsilon), followed by 30 tanh(logit/30). These equations therefore predict finite, simultaneous source edits exactly over this fixed-state interface, rather than approximating their effects with a derivative. The Gram matrix is shared by all readers, while each reader has its own numerator. Cross-terms between sources must be retained for joint edits.

The [executed CPU implementation](../rms_softcap_edit_statistics_v1.py) reproduced direct decoding within 8.89e−16 and orthogonal-coordinate invariance within 3.22e−15. It preserved the native FP32 epsilon when evaluating in FP64, handled exact state cancellation, and demonstrated a 0.149 output error when source cross-terms were deliberately omitted. These are known decoder identities made executable, not discovered learned sharing.

A concrete counterexample explains the failed vector ranking. Let a large source be (0,100), a small source (1,0), and the reader see only the first coordinate. The large source accounts for 99.99% of the total vector's signed projection, but removing the small source eliminates the reader output. Removing the large source instead increases it by changing normalization. Large vector contribution can have the wrong relationship to behavioral importance.

The two completed native removal reports also constrain the remaining attention contribution. Subtracting the all-MLP removal effect from the total mixed-state removal effect gives **57.87–97.56%** signed projection for removing the remaining attention component *after the MLP component is already absent*, up to audited numerical closure. This is conditional evidence, not an attention-only intervention in the native background. That attention-only test has since run; section 9 reports its outcome. No attention head is selected from these figures.

The next native measurement should retain these reader and norm statistics and test the attention contribution in both backgrounds before naming its producer. This prevents another source-size ranking from masquerading as circuit localization. [CPU controls and conditional diagnostic](../RMS_SOFTCAP_EDIT_STATISTICS_V1_CONTROLS.json). All native weights and contextual producers remain charged; independent extraction, new OOD prediction, selective unrelated-behavior removal, and reusable circuit composition are still incomplete.


## 9. Attention contributes substantially; joint direct-write edits compose

The [registered attention/MLP factorial](../THIRD_NOUN_ATTENTION_READER_V1_PREREGISTRATION.md) decoded the final native state with neither, either, or both sets of transported mixed writes removed. Attention-only removal leaves **26.88–43.28%** of the original answer-interaction magnitude, failing the registered 25% maximum in every world. Its signed removal projection is 57.86–97.55%; a large signed projection does not ensure the remaining vector is small.

Removing both attention and MLP mixed writes leaves 7.08–11.11%, reproducing the earlier whole-state result. The two removal effects compose almost additively: the mixed part of

\[
y_{AM}-y_A-y_M+y_0
\]

has only **0.0062–0.0139%** of the original mixed answer effect's magnitude. This passes the prospective composition bar. Full-vocabulary mixed nonadditivity is also small, 0.0029–0.0254%. These are compositions of fixed native-write edits at the final residual interface, not independently extracted circuits or upstream interventions with downstream recomputation.

The run validated the exact reader/norm statistics against all four full-vocabulary GPU decodes. Maximum two-answer discrepancy was 3.10e−6 logits, relative error at most 1.09e−7. Native parent replay was exact; the MLP arm replayed exactly and the joint-removal ratio differed from the earlier test by at most 1.18e−6. Executor cost: 16 forwards / 256 prefixes, 64 decoder batches / 1,024 states, 0.961 seconds, no fitting. [Native receipt](../THIRD_NOUN_ATTENTION_READER_V1_RESULT.json).

## 10. Reader-weighted layer localization, with no additional model calls

The run saved actual reader projections and shared norm statistics for each of 18 attention-layer writes plus the aggregate MLP write. The subsequent [CPU audit](../third_noun_reader_route_audit_v1.py) reused these to evaluate individual attention-layer edits and to separate numerator changes from denominator changes in the final decoder.

A numerator-only diagnostic changes the two answer projections while keeping the native normalization denominator; a denominator-only diagnostic does the reverse. These diagnostic combinations need not correspond to a physical residual vector. The denominator-only effect is **0.0566–0.1550%** of the original mixed answer effect's magnitude. Thus the substantial attention effect principally changes answer evidence, not a global normalization scale.

Layer 9 attention is the largest individual signed contributor in every one of the eight worlds, at **18.21–38.06%** of the original mixed answer effect. Layers 10 and 12 are often next, but their order depends on the lexical world. This is a distributed computation: layer 9 alone is not a sufficient circuit. The sum of individual attention-layer effects matches their joint effect to within 0.00565% of the original mixed answer magnitude on this frozen interface.

The next informative native question is how the relevant attention writes are formed: their query/key routing, value content, and upstream producers. Layer 9 supplies a candidate within-module interface for that question. It was localized from opened data; this is not fresh OOD identification, and no particular head, token edge, or rank has been chosen or registered yet. Shared use of layer 9 by another task would not by itself establish the same reusable computation.

[CPU audit receipt](../THIRD_NOUN_READER_ROUTE_AUDIT_V1_RESULT.json). All native weights and input-producing computations remain retained and charged. Neither attention nor MLP alone passes the registered sufficiency bar; preserve those failures. The positive composition result applies to synthetic final-state source edits, while independent extraction, selective unrelated-behavior removal, new OOD semantic prediction, and reusable learned-program composition remain incomplete.
