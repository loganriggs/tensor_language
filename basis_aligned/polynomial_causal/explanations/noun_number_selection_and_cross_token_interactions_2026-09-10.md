# Noun-number selection: what the model does, and what the math rules out

**Latest result:** the four-combination test separates MLP8’s effect into attention9’s response and routes bypassing that response. Both are needed; their interaction is small in all 32 groups. The bypass has the larger signed contribution, so the next question is whether it is direct residual carry or depends on later computations. Section 23 gives the native evidence and the exact transport control.

The new tests move beyond the stagnant is/was investigation. We first specified a simple computation—choose the noun whose number controls a reflexive—and checked whether the model actually performs it. It does not reliably switch the controlling noun with the verb. In short two-noun sentences, all 128 measured preferences follow the second noun. Adding a third noun then breaks each of four simple rules we had registered in advance.

The useful mathematical lead is **context-dependent combination of noun-number signals**. Making the third noun human reduces the second noun's influence and increases the third noun's influence. We have verified that the former interaction cannot originate in a token-local lookup. A subsequent native intervention now shows that most of the answer interaction is carried in the internal residual state: removing it leaves only 7.1–11.1% of its original magnitude. Raw-vector accounting favored MLP writes, but the subsequent behavioral test rejected them as the main answer carrier: removing all their mixed writes leaves 68–110% of the answer interaction. The subsequent attention test also fails sufficiency alone, leaving 27–43%, while attention and MLP removals compose almost additively. Reader-weighted localization identifies layer 9 attention as the largest individual contributor in every world, primarily through the answer logits rather than normalization. The latest exact routing/value split needs multiple terms. Causal source support distinguishes reading earlier number-bearing positions from reading later contextual positions, and proves that the already-mixed value term comes from the local contextual value map. The latest live-interface test establishes a partial causal contribution: removing the local mixed value changes 16–25% of the natural interaction while largely preserving the other factor components. Its downstream response cannot be replaced by direct residual carry alone. Independent extraction and the full four-property goal remain incomplete.

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


## 11. Layer9 needs multiple routing/value terms; causal support distinguishes their inputs

For a head and source position, let P be its native routing weight and V its native mixed value vector. P is the product of the two normalized, rotary-positioned query/key dot products; it is not a softmax probability. The exact conditional interaction is

\[
(PV)_{oh}=P_{oh}V_0+P_0V_{oh}+P_oV_h+P_hV_o.
\]

The four coefficients are defined by symmetric averaging over the object-number/category square while keeping the other factors fixed. They can already contain upstream nonlinear computation. We grouped the terms into routing-inherited, value-inherited, and routing–value cross branches, summed every head/source, and applied the actual output projection and residual carry.

The [native test](../THIRD_NOUN_L9_ROUTE_VALUE_V1_RESULT.json) confirms the exact factorization and layer9 materiality, but **none of the three branches passes the registered 10% local-effect error bar**. Their answer-effect errors relative to the complete layer9 contribution are:

| Branch used alone | Relative answer-effect error |
|---|---:|
| Routing-inherited | 0.919–1.127 |
| Value-inherited | 0.372–0.760 |
| Routing–value cross | 0.340–1.055 |

The full-vector fidelity tests fail too. Thus no single branch should be promoted, and the bar was not relaxed. The layer9 contribution remains 18.21–38.06% of the natural mixed answer effect. This local decomposition does not explain the entire model behavior.

Algebraic closure error was 1.40e−15 relative; agreement with the actual native mixed write was 3.51e−6. Read-only captures retained native normalization, RoPE, shared-value mixing and upstream producers. Cost was 16 forwards / 256 prefixes and 80 decoder batches / 1,280 final states in 0.943 seconds. [Protocol](../THIRD_NOUN_L9_ROUTE_VALUE_V1_PREREGISTRATION.md), [partition code](../attention_route_value_partition_v1.py), and [controls](../ATTENTION_ROUTE_VALUE_PARTITION_V1_CONTROLS.json).

Causal order gives a more informative split than picking the largest branch. Using zero-based positions, object number changes token 4 and human/inanimate category changes token 7:

- At positions 0–3, values can know neither edited factor. Their value and cross terms vanish.
- At positions 4–6, values can know object number but cannot know the later category. Therefore V_h=V_oh=0 and the cross term reduces to **P_h V_o**: category-dependent routing of earlier number-bearing information.
- At positions 7–9, values can know both factors. Both inherited mixed values and the two cross terms are possible.

The category's influence on routing to positions 4–6 must enter from the later query side; the earlier keys cannot depend on a future category token. This does not make P_h a standalone category detector: it is a finite-table component of a contextual routing computation.

The [executed CPU source audit](../THIRD_NOUN_CAUSAL_SOURCE_SPLIT_V1_RESULT.json) verifies both predicted early-source zeros exactly in the saved reader numerators. In fixed-native-normalization, pre-softcap reader accounting, the earlier-number cross branch has signed projection −0.155 to 0.549 onto layer9's complete linear effect. The later-context value branch has 0.294 to 1.046. Negative or greater-than-one values reflect cancellation. These are diagnostic linear contributions, not physical single-edge removals or successful standalone circuits.

Head6 has the largest signed contribution in seven worlds, head7 in the remaining world, under that same diagnostic. This does not license a universal head6 circuit. More importantly, the causal split identifies different information dependencies within a native module without assuming that head boundaries are semantic boundaries.

There is an exact weight consequence for the contextual value branch. Native values are

\[
V_s=(1-\lambda)W_Vu_s+\lambda V^{(1)}_s,
\]

where u_s is the current normalized contextual input and V^(1) is the shared first-layer value. The first-layer value is a function of the token at its own position, so its oh component vanishes on this disjoint-edit domain. Hence

\[
(V_s)_{oh}=(1-\lambda)W_V(u_s)_{oh}.
\]

For each head, W_O can be folded with W_V when evaluating this specific branch, while the contextual normalized input and routing coefficient P_0 remain explicit. This identifies where its mixed value must be produced; it does not remove the upstream computation or its normalization.

A CPU checkpoint read confirms layer9 lambda=**−0.65625**, so the local coefficient is **1.65625**. The native combination adds an amplified local value and subtracts a shared value; the coefficients are not probabilities. The shared value can still contribute to other terms such as earlier-number routing. [Coefficient receipt](../THIRD_NOUN_L9_VALUE_MIX_COEFFICIENT_V1.json).

All native weights remain charged, and no head/source/rank is promoted from these diagnostics. The next useful target is the producer of the later contextual mixed value, or a full producer-level test of the earlier-number routing term. Neither next native experiment has been registered or queued yet. The post-result causal-source audit and checkpoint coefficient read have actually been executed; no extra native forward was needed.


## 12. Live local-value removal establishes a partial causal component

The previous source edits changed the final residual with other native writes held fixed. We have now intervened at the actual layer9 value projection and let the model recompute the rest of layer9 and every subsequent layer.

From the observed normalized input u, we compiled the local value correction **W_V P_oh u** using the checkpoint's actual value weights, then subtracted it from the local c_v output. The native value-mixing coefficient was applied normally. Queries, keys, second queries/keys, and the shared first-layer value stayed unchanged. Every head and position was included; causal support made the correction zero at positions before7.

The [registered live-interface test](../THIRD_NOUN_L9_VALUE_LIVE_V1_PREREGISTRATION.md) passed instrument validity, partial causal materiality, and factor selectivity:

- The actual output effect's signed projection onto the natural mixed answer effect is **15.70–25.33%**, across all eight worlds.
- Other answer-response factor components have **13.73–23.31%** of the target mixed component's magnitude, below the registered25% ceiling. These include main effects and other interactions within this same sentence bank; they are not unrelated-task controls.
- The native interaction remaining after removal is **74.89–84.40%** by magnitude. This is a partial component, not a sufficient circuit explaining the full behavior.

A direct-residual-only prediction fails: its error relative to the live intervention is 9.50–37.43% for the mixed answer effect and 85.83–92.03% for the full mixed vocabulary vector. Later computations respond to the value edit and must remain part of any faithful extracted program. The failure of full-vector prediction is not a claim that all of that error matters to this particular answer contrast.

Identity replacement was exact; Q/K/Q2/K2 and shared firstV remained bitwise unchanged. Weight commutation error was 4.74e−6 relative, and the native local-write oracle error at most1.66e−5. Native parent replay was exact. The saved-final decoder replay passed both registered tolerances, with maximum absolute discrepancy2.67e−5 and relative error4.62e−7.

Execution used 48 transformer forwards over768 sequence instances—baseline, identity and intervention—plus32 decoder batches over512 states including the native-final replay, in2.015 seconds. No fit, neuron selection or rank search occurred. All545902902parameters remain charged, with structural savings0. [Native result](../THIRD_NOUN_L9_VALUE_LIVE_V1_RESULT.json).

The correction still requires four native inputs per fixed context. It has not been produced independently from one sentence, and it has not yet transferred to a new lexical bank or a different reflexive answer reader. Those are the next generalization/reuse questions before further producer decomposition.

The post-result [shared effect scorer](../factorial_effect_metrics_v1.py) reproduced the existing metrics within2.78e−17 and verified the exact finite-table Parseval identity. Its [audit](../LIVE_VALUE_FACTOR_METRICS_V1_RESULT.json) shows that the largest remaining non-target term varies by world: object number, human category, verb×object number, or object×third-noun number. There is no single spill term to remove uniformly. This is descriptive analysis of opened cases, not a tuned correction.

The [06:14 hourly review](../HOURLY_STRATEGIC_REVIEW_2026-09-10_0614.md) records five native screens and a9m36 median consecutive-receipt interval. Whole-session validation/reporting overhead remains unmeasured; the small shared scorer and its timed CPU analysis/validation stages are a bounded reuse repair, not proof that the whole overhead budget is met. Next reviews are07:14 hourly and07:49 mathematical. The fresh-transfer test has now completed, as described below.

## 13. Fresh nouns preserve the causal effect, but expose failed selectivity

We froze the local-value operation and tested 512 fresh prefixes: husband, gentleman, monk, dad, queen, woman, girl and mother, each with defend and introduce, across the same 32 combinations of sentence factors. Male groups use themselves versus himself; female groups use themselves versus herself. The human attractor rotates within each gender panel, and the inanimate alternatives are rock, lamp, chair and desk. This tests a new lexical bank and another answer reader within the same sentence construction; it does not establish arbitrary structural OOD generalization.

The [registered test](../THIRD_NOUN_VALUE_TRANSFER_V1_PREREGISTRATION.md) completed validly. Native and edited reference outputs replayed exactly, queries/keys/shared first values were preserved, and the run used 70 forwards over 1120 sequence instances in 1.328 seconds, with no fitting.

Every fresh group has a live native interaction and passes the materiality requirement: the intervention's signed projection onto the native interaction is **12.86–26.89%**. But the requirement that unwanted factor effects stay below 25% of the target effect fails in six groups: monk/defend, dad/both actions, and queen/girl/mother with introduce. Each gender panel therefore passes the complete transfer test in only **5 of 8** groups. The failure is selectivity, not disappearance of the target effect. We retain the original thresholds, including the close misses. [Full result](../THIRD_NOUN_VALUE_TRANSFER_V1_RESULT.json).

To check whether the two answer readers really isolate number, let the three actual output logits be z_t, z_m and z_f for themselves, himself and herself. Define

\[
N=z_t-(z_m+z_f)/2,\qquad G=z_m-z_f.
\]

N measures plural versus average singular preference; G measures the difference between the singular gender answers. The two reader coefficient vectors are orthogonal. Their answer margins satisfy exactly

\[
z_t-z_m=N-G/2,\qquad z_t-z_f=N+G/2.
\]

The registered gender-effect/number-effect norm ratio must be at most .25 in every group. It fails for monk/introduce (**.3985**) and woman/introduce (**.3003**). Thus use of both answer readers does not establish a gender-independent hidden variable.

The subsequent [CPU geometry audit](../THIRD_NOUN_READER_GEOMETRY_V1_RESULT.json) verifies these identities and the centered three-logit energy identity

\[
\|z-\bar z\mathbf1\|^2=\tfrac23 N^2+\tfrac12 G^2.
\]

Applied to the mixed intervention effects, number accounts for 89.36–99.11% of this energy, and the two answer-effect vectors have cosine .9817–.9983. These attractive similarities coexist with failed selectivity. Energy concentration and similar outputs cannot substitute for the intervention control, or prove that two consumers share an identified internal computation.

**Why the input decomposition does not guarantee selective output behavior.** Let o and h be the ±1 object-number and human-category factors. Even a perfectly isolated input correction d=oh can be read by a context-dependent multiplier J=1+βh. Its output effect is

\[
Jd=(1+\beta h)oh=oh+\beta o,
\]

because h²=1. A pure input interaction creates an unwanted object-number main effect. The [executed four-corner control](../CONTEXTUAL_READER_FACTOR_SPILL_V1_CONTROL.json), with β=.3, gives exactly 30% spill. This is a counterexample to automatic selectivity, not proof that this multiplier explains the native model.

More generally, the live suffix's finite response is an integral of its Jacobian along the removed-value path. Both the input correction and that reader depend on sentence context. A projector over the input sentence table therefore need not commute with the downstream computation. A useful weight-based split must account for the producer **and its consumers**; merely making the input factors orthogonal is insufficient.

The next mathematical target is to distinguish context-dependent downstream reading from a mixed producer that bundles different variables. This calls for a discriminating reader/producer test, not threshold changes or removal of the six failing groups. The CPU geometry audit and toy falsifier are completed continuation work; no successor GPU experiment is registered yet. Independent production still requires replacement of the four native counterfactual input states, and all 545902902 native parameters remain charged.

## 14. Splitting the attention write fixes factor selectivity, but not gender selectivity

The next test intervened at the output of layer 9 attention. Let D be the complete change in its write caused by the original local-value removal, measured at every position. We split it into D_m=Q_oh D, the intended interaction, and D_s=D-D_m, the other factors. Here Q_oh is the same four-corner projector previously denoted P_oh; Q distinguishes it from attention routing below.

We removed D_m alone, D_s alone, and both together, with the native downstream layers recomputing. Removing both exactly recreates the original local-value edit at this interface, providing a positive control. The [registered factorial](../THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_PREREGISTRATION.md) passes instrument validity, mixed-effect fidelity, factor selectivity and composition. It fails gender selectivity:

| Measurement | Result across all 16 groups |
| --- | --- |
| Intended mixed answer effect: difference from the original intervention | 0.65–9.23% relative error; all below 10% |
| Other factor effects / intended effect, after removing D_m alone | 6.39–22.37%; all below 25% |
| D_m effect's signed projection onto the natural interaction | 13.91–27.12%; still a partial contribution |
| Joint removal versus sum of separate effects | At most 1.46% error for the correct margin and 1.34% for the centered three-answer vector |
| Gender effect / number effect | Fails for monk/introduce (.3961) and woman/introduce (.2943) |

The full-write replay and both parent-output replays pass the registered absolute and relative bounds. The run used 160 transformer forwards over 2560 sequence instances in 2.535 seconds. All three write patches cover all positions, preserve the shared first-layer value, and use the original native suffix. [Native result](../THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json).

This locates a removable source of unwanted factor effects in the way attention reads the value correction. It does not show that later layers produce no unwanted effects: D_m alone still has nonzero spill. It also does not make all of D_s unimportant; the fidelity test permits up to 10% difference in the target effect.

There is a useful exact interpretation in the bilinear attention algebra. Fix the other sentence factors and write the pure value correction as d(o,h)=oh d_oh. Let P(o,h) denote the complete linear map from these value corrections to attention writes, with the native queries, keys, value-mixing coefficient and output weights included. Expand that map as

\[
P(o,h)=P_0+oP_o+hP_h+ohP_{oh}.
\]

Then

\[
P(o,h)d(o,h)=ohP_0d_{oh}+hP_od_{oh}+oP_hd_{oh}+P_{oh}d_{oh}.
\]

Consequently, **D_m=P_0d**: the selected write reads the mixed value through routing averaged over the four o/h corners. The remaining terms describe routing-dependent conversion into other factors. This is a conditional identity; P_0 may still depend on the verb, other noun numbers, lexical world and positions. It does not mean attention routing is globally constant or that we can discard the producer of d.

The [executed CPU audit](../THIRD_NOUN_WRITE_FACTOR_ATTRIBUTION_V1_RESULT.json) verifies the matrix-valued identity to 4.44e−16 and attributes the saved unwanted output effects to the two edits and their interaction. Removing only D_m leaves 31.5–95.2% of the original unwanted-effect magnitude, depending on the group. Signed contributions reveal cancellation: the D_s contribution can exceed 100% where D_m partially opposes it. Therefore these components should not be presented as a simple positive percentage partition.

This is the kind of within-module split the original handoff calls for: one native attention module contains separable contributions with different downstream roles. However, its production still uses native counterfactual inputs; its gender control fails; and its apparent factor-selectivity improvement is measured on an already opened bank. The next useful test is to freeze this exact split and test structural OOD transfer, before claiming a reusable extracted unit. All 545902902 native parameters remain charged. The post-result CPU attribution and routing identity are completed; no structural-transfer GPU job is registered yet.

## 15. Longer phrases transfer; moving the category cue to the front does not

We froze the mixed-write split and tested two new structures with the same nouns and actions:

- “The husband persuaded the husbands **in the room near the lamp** to defend …”
- “**Near the lamp,** the husband persuaded the husbands to defend …”

Each structure has 16 noun/action groups and 32 counterfactual combinations per group: 1024 new prefixes total. The longer structure has 13 tokens; the fronted structure has 11. Within every group, paired edits retain equal lengths, and opposite subject/object-number assignments retain the same token multiset. The fronted structure changes both syntactic arrangement and evidence order: the category cue now precedes the object. This test does not isolate those two changes from each other.

The generalized executor first reproduced the original reference's baseline and all three intervention output grids exactly. It then used the same layer, weights, all-position intervention and thresholds on the new bank. The [registered structural test](../THIRD_NOUN_WRITE_STRUCTURE_V1_PREREGISTRATION.md) is mechanically valid, with 266 forwards over 4256 sequence instances in 4.382 seconds.

| Registered requirement | Longer intervening phrase | Fronted phrase |
| --- | --- | --- |
| Live native interaction, RMS at least .05 | 16/16 | 16/16 |
| Partial materiality and factor selectivity together | 16/16 | 7/16 |
| Mixed-write effect within 10% of the original local-value effect | 16/16 | 16/16 |
| Gender effect below 25% of number effect | 13/16 | 3/16 |
| Two edits compose within the registered tolerance | 16/16 | 16/16 |

The all-world structural-transfer and gender predictions therefore **fail**. The result supports the longer-layout branch under the unchanged criteria, without rescuing the failed fronted branch. Fronting yields eight materiality failures and five factor-selectivity failures, with overlap; all native interactions remain above the registered floor. [Full result](../THIRD_NOUN_WRITE_STRUCTURE_V1_RESULT.json).

The subsequent [paired geometry audit](../THIRD_NOUN_STRUCTURE_GEOMETRY_V1_RESULT.json) separates effect size from direction. Let n be the native mixed answer vector across the 32 cases and e the mixed intervention effect. The reported signed projection is exactly

\[
\frac{e^\mathsf T n}{\|n\|^2}
=\frac{\|e\|}{\|n\|}\cos(e,n).
\]

In the longer layout, the effect aligns closely with the native interaction: cosine .9677–.9953. In the fronted layout it ranges from −.3495 to .9815. Four groups have a negative projection: removing the component changes the answer interaction in the opposite direction to the native interaction. These effects are not absent; their effect/native norm ratios exceed .05. A positive gain cannot repair the negative alignment, because multiplying e by a positive scalar preserves its direction.

Comparison with the original lexical bank also shows that eight fronted intervention-effect vectors have negative cosine with their original counterparts, while every native-interaction vector retains positive cosine. Thus the operation's behavioral role changes substantially across structures. This output-space diagnosis does not yet say whether the change originates in the value producer, averaged routing, or later readers.

The useful distinction is now clearer: **the algebraic split transfers more reliably than its proposed semantic role**. Faithful decomposition and predictable joint edits do not by themselves identify a shared number computation. The next discriminating question is where the structural change alters that role; another gain, threshold, or row selection would not answer it. The paired CPU audit is completed continuation work. No further native experiment is registered yet, and independent extraction plus the full four-property goal remain open.

## 16. A common causal interface at the final two positions

Before exchanging writes between structures, we checked when each token position can know both factors. In the original layout, the object comes before the attractor; the first possible joint interaction is at the attractor. In the fronted layout, the attractor comes before the object; the first possible interaction is at the object.

Consequently, mapping the whole write by noun role would be a poor test of reusable causal computation. An original attractor-position write may contain object-number information, but the fronted attractor position has not seen that object yet. The reverse map has the same problem at the object. Such patches can be executed, but they can inject future information; agreement would not establish a naturally available shared computation. The [CPU role audit](../THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1.json) checks this for all 16 paired worlds.

The shared roles that have seen both cues in either layout are **“to” and the final action word**. We therefore divided the mixed attention write into C, retaining these two positions, and N, retaining every other position. Causal support limits N to the noun where the second factor becomes available. We tested removing C, N and their sum, with the native suffix recomputed. This division was fixed from role and causal information, without searching positions by their measured effects.

The [registered common-suffix test](../THIRD_NOUN_COMMON_SUFFIX_V1_PREREGISTRATION.md) passes instrumentation, common-interface fidelity and composition:

- C alone reproduces the complete mixed component's answer effect with **0.21–3.93% relative error**, across all 32 original/fronted worlds.
- N alone fails the same fidelity test in every world: **98.53–101.90% relative error**.
- The separate effects compose with at most **0.094%** error for the correct margin and **0.106%** for the centered three-answer vector.
- Original and fronted baseline/full-component output grids replay exactly. The run used 320 forwards over 5120 sequence instances in 4.874 seconds.

This is sufficiency relative to the previously identified **partial intervention effect**, not sufficiency for the model's entire natural interaction. The common write still passes factor selectivity in only 10/16 fronted groups, versus 16/16 original groups; gender control passes 3/16 fronted and 14/16 original groups. The localization does not erase those failures. [Native result](../THIRD_NOUN_COMMON_SUFFIX_V1_RESULT.json).

We can now pose the producer-versus-reader question without the earlier role-mapping defect. Hold the recipient layout fixed and supply the common write from either layout, at the matching “to” and action positions. Both donor and recipient positions have already seen both cues, and the residual-vector coordinates are those of the same checkpoint.

Let E_ij be the causal answer effect when the write comes from layout i and the surrounding recipient computation is layout j. Here “producer” includes the computation of the selected layer 9 write; “reader” includes its recipient background and later layers. Two opposing predictions are:

- **Effect follows the producer:** E_10 resembles E_11, and E_01 resembles E_00.
- **Effect follows the reader:** E_10 resembles E_00, and E_01 resembles E_11.

Neither outcome is guaranteed. The interaction E_11−E_10−E_01+E_00 can make both simple descriptions fail. The [new CPU accounting core](../cross_context_effect_attribution_v1.py) correctly distinguishes planted producer-only, reader-only and interacting examples. Its [control receipt](../COMMON_SUFFIX_CONTINUATION_V1.json) also summarizes the completed native localization. No native cross-layout interchange has been run or registered yet; these are prospective predictions, not findings. Independent extraction and all remaining native weights are still unresolved.

## 17. Interchange points toward the write producer, with a remaining reader contribution

We performed genuine component replacements in both directions. For recipient layout r, define B_r=A_r−C_r: its native layer 9 attention write with its own common component removed. A donor component C_p is then installed as B_r+T C_p, where T maps the two matching token roles. The measured causal effect is

\[
E_{pr}=z_r(B_r+TC_p)-z_r(B_r).
\]

Thus the donor is tested against a fixed recipient background. We did not merely subtract an unrelated donor write from the unmodified recipient. All 16 lexical/action pairs, both directions, the original weights and native suffix were retained, with no fitted alignment or gain.

The [registered interchange](../THIRD_NOUN_COMMON_INTERCHANGE_V1_PREREGISTRATION.md) is valid. Native and removed-component outputs replay exactly. It used 256 forwards over 4096 sequence instances in 4.274 seconds. The strict producer-following and reader-following predictions both fail; the small-interaction prediction passes in every pair. Each following rule required errors below10% in both directions for both the correct mixed answer margin and the centered three-reader mixed vector.

For the correct answer margin, producer-following errors range from **3.67–24.35%**, whereas reader-following errors range from **61.99–909.43%**. These have the registered direction-specific diagonal effect as denominator; the larger percentages can reflect comparison with a small diagonal effect. The interaction is at most **9.09%** of the larger diagonal-effect norm for the margin and **9.84%** for the centered three-reader vector. Smaller producer-following errors do not turn its failed universal prediction into a pass. [Native result](../THIRD_NOUN_COMMON_INTERCHANGE_V1_RESULT.json).

To describe the measured structural change without discarding the reader contribution, the [CPU attribution](../COMMON_INTERCHANGE_CHANGE_ATTRIBUTION_V1_RESULT.json) uses the exact symmetric decomposition

\[
\Delta E=E_{11}-E_{00}=\Delta_P+\Delta_R,
\]
\[
\Delta_P=\tfrac12[(E_{10}-E_{00})+(E_{11}-E_{01})],\qquad
\Delta_R=\tfrac12[(E_{01}-E_{00})+(E_{11}-E_{10})].
\]

The producer term's signed projection onto the observed margin change is **90.62–103.25%**. The reader term ranges from **−3.25% to9.38%**, with reader-change magnitude1.94–11.05% of the total change. Values above100% or below0 indicate cancellation. This is an attribution of this measured two-context contrast, not proof that readers never matter or that the producer executes independently.

There is now a bounded weight-based object to investigate inside that producer. In both layouts, a mixed value can first occur at the third-to-last token. The three source stages are: the token where both cues first become available, “to,” and the action. The output queries are the final two stages. Their causal pattern has the same shape:

\[
\begin{pmatrix}1&1&0\\1&1&1\end{pmatrix}.
\]

The zero prevents “to” from reading the future action token. The first source is an attractor in one layout and an object in the other: this is alignment by information availability, not an assertion that the noun roles are identical.

For these positions the ideal mixed-write contraction is

\[
C_q=(1-\lambda)\sum_{h,s}(P_0)_{hqs}\,
W_{O,h}W_{V,h}(u_s)_{oh}.
\]

Here u_s is the actual normalized contextual input, (u_s)_oh its mixed component, P_0 the conditional mean of native attention routing, and W_V/W_O the checkpoint's value/output maps for head h. The local coefficient is1−λ=1.65625. This formula retains the normalized producers and position-dependent routing; it permits folding the adjacent linear maps without treating that rewrite as interpretability by itself.

The [new contraction helper](../mature_value_route_contract_v1.py) and [CPU controls](../MATURE_VALUE_ROUTE_CONTRACT_V1_CONTROLS.json) verify the conditional routing identity to1.33e−15 and check the shared causal shape in all32 native row geometries. A native value-versus-routing factorial can now test the two inputs to this contraction while retaining both recipient contexts. It has not yet been registered or run. No independent extraction, new structural-transfer success, or reduction in opaque native weights is claimed.

## 18. The weight contraction is faithful; values and routing remain coupled

We ran the full value × routing × recipient experiment: each factor takes the original or fronted context, giving eight combinations per lexical/action pair. The value factor uses the actual W_V projection of the mixed normalized input; the routing factor uses the conditional mean of the native product-attention kernel after its normalizations and positional rotations. W_O and the local coefficient1.65625 finish the write. The recipient's native background and suffix remain explicit.

The [registered native test](../MATURE_VALUE_ROUTE_V1_PREREGISTRATION.md) passes its instrument checks. The compiled common write agrees with the observed component to at most **1.90e−5 relative error**. Native, removed and homogeneous-producer interchange outputs replay within **8.59e−6 absolute** and **2.88e−7 relative** error. Only after those checks were the hybrid producers interpreted. Execution used448forwards over7168sequence instances in7.184seconds. [Native result](../MATURE_VALUE_ROUTE_V1_RESULT.json).

The scientific predictions fail:

- Neither **value-following** nor **routing-following** passes any of the16pairs across both recipients and both readout objects.
- Correct-margin value-following errors range from4.38–65.16%; routing-following errors range from63.23–909.71%, with the registered direction-specific denominator.
- The small-interaction requirement passes only dad/defend. Value–routing interaction reaches64.67% of the larger diagonal-effect norm for the correct margin and53.10% for the centered three-reader vector.

Thus faithful factorization does not make the factors independently reusable semantic units. Even at the write interface, a bilinear contraction L obeys

\[
L(P_1,V_1)-L(P_1,V_0)-L(P_0,V_1)+L(P_0,V_0)
=L(P_1-P_0,V_1-V_0).
\]

Changing both factors can produce a contribution absent from either separate change. This identity is at the attention-write interface; the live suffix can further change the resulting output effects. Our native test measured those effects rather than assuming the suffix was linear.

The [post-result CPU audit](../MATURE_FACTOR_PATH_ATTRIBUTION_V1_RESULT.json) averages the six possible orders of changing value, routing and recipient. Each path telescopes from the all-original effect to the all-fronted effect, so the three averaged contributions sum exactly to the observed structural change. Their signed projections onto the correct-margin change are81.33–124.20% for values,−22.02–14.68% for routing, and−3.19–9.56% for recipient context. These are path-averaged signed attributions, with interactions distributed among the factors; they do not remove the interaction or rescue the failed one-factor rules.

This directs the next producer question toward the normalized contextual value while keeping its routing partner. One useful distinction is whether the mixed value already exists in the incoming residual vector or is generated by its normalization. That can be tested with the existing factor algebra, retaining the native joint routing. No such native test is registered yet. The contraction still requires native counterfactual inputs and all original weights; it is a faithful local program, not an independently extracted semantic circuit.

The [07:14 hourly review](../HOURLY_STRATEGIC_REVIEW_2026-09-10_0714.md) records five prior tests, median successive completion interval8m10.5 and a leading21m45 gap. Preparation/review time was not retrospectively measurable. A reusable phase clock now records the current block prospectively; this does not establish a past overhead budget. Next hourly review08:14 UTC; mathematical review07:49 UTC.

## 19. Normalization is small in the original layout, but cannot be dropped across structures

The input to the value map is u=gx, where x is the raw residual entering layer9 attention and

\[
g=\left(\operatorname{mean}(x^2)+\epsilon\right)^{-1/2}.
\]

We retain the checkpoint's FP32 epsilon explicitly. The same conditional factor algebra gives

\[
u_{oh}=\underbrace{g_0x_{oh}}_{R}
+\underbrace{g_{oh}x_0+g_ox_h+g_hx_o}_{N}.
\]

R carries a mixed signal that already exists in the raw vector, through the conditional mean scale. N contains mixed terms generated by changes in the normalization scale. We mapped R, N and their sum through the same native value/routing/output contraction, removed the resulting common writes separately, and let the native suffix recompute. No raw-residual edit was made that would also alter queries and keys.

The [registered test](../MATURE_VALUE_NORMALIZATION_V1_PREREGISTRATION.md) is valid and its two branch effects compose within the registered tolerance in all32worlds. The raw recurrence reproduces native RMS input bitwise; FP64 normalized-input reconstruction differs by at most5.55e−8 relative. Execution used320forwards over5120sequence instances in5.147seconds.

| Branch reproducing the full partial-component effect within10% on both readouts | Original layout | Fronted layout |
| --- | --- | --- |
| Inherited raw-state branch R | 16/16 | 6/16 |
| Normalization-generated branch N | 0/16 | 0/16 |

The universal raw-origin and normalization-origin predictions both fail. In original cases, raw-branch error is at most3.69%. In fronted cases it reaches62.11% for the correct margin and57.01% for the centered three-reader vector. The normalization-only failure does not mean normalization is irrelevant. [Native result](../MATURE_VALUE_NORMALIZATION_V1_RESULT.json).

The [post-result signed accounting](../NORMALIZED_VALUE_BRANCH_ACCOUNTING_V1_RESULT.json) makes this distinction explicit. In the original layout, the normalization branch's effect magnitude is1.19–3.66% of the full component's effect; in the fronted layout it is2.25–62.12%. Its signed projection ranges from−2.35% to2.40% in the original layout and−10.53% to23.19% when fronted. It can oppose the raw contribution or point partly in another direction. The denominator is the previously identified partial component, not the model's entire behavior.

This is a decomposition of the value producer with routing held fixed. Removing R is not the same as removing x_oh before RMS: the latter also changes g. A genuine upstream intervention must recompute the complete normalization and account for that response. The next producer trace should retain this known primitive and test raw-state changes through it, rather than approximate it away or infer causal source importance from raw vector norms. No successor native test has yet been registered. Independent state production, structural semantic generality, and the full four-property goal remain unresolved.

## 20. Actual raw-input removal and a small exact normalization interface

We now performed the intervention that section 19 distinguished from attribution. Subtract the raw mixed component from the layer 9 attention input, recompute native RMS normalization, rebuild the partial value contribution with the original routing, and install it in the original background. The remaining model then recomputes normally. This changes the value-producing path; queries and keys retain their native inputs.

The [registered native test](../MATURE_VALUE_RAW_REMOVAL_V1_RESULT.json) passes its implementation checks. Native and full-component-removal outputs replay exactly; reconstructing the unchanged component differs from native answer logits by at most 8.59e-6. It used 320 forwards over 5,120 sequence instances, taking 5.01 seconds inside the executor.

Raw-input removal matches full-component removal within 10% on both measured output objects in **16/16 original groups and 5/16 fronted groups**. The worst original error is 3.81%; fronted errors reach 68.00% on the correct margin and 62.53% on the centered three-answer vector. Neither universal prediction passes: raw inheritance does not explain every case, and normalization alone does not reproduce the component in any group. These denominators concern the identified partial component, not the model's entire behavior. The earlier 6/16 count concerned removing an algebraic branch, a different intervention.

There is a useful exact mathematical consequence. Holding the other factors fixed, the residual after removing its mixed term has the form

\[
x'(o,h)=a+ob+hc,\qquad o,h\in\{-1,+1\}.
\]

Here a is the mean vector, b the object-number coefficient, and c the attractor-kind coefficient. Its squared RMS is

\[
\frac{\|x'\|^2}{d}
=\frac{\|a\|^2+\|b\|^2+\|c\|^2
+2o\langle a,b\rangle+2h\langle a,c\rangle
+2oh\langle b,c\rangle}{d}.
\]

Thus **six inner products determine the normalization scales** at all four corners. A Gram matrix is simply the table of those pairwise inner products. Applying the inverse square root to the four scalar norms gives four scale coefficients. The normalized mixed coefficient is exactly

\[
(RMS(x'))_{oh}=g_{oh}a+g_hb+g_oc.
\]

A downstream linear reader W can be folded into the three vectors: the answer is the same scalar combination of Wa, Wb and Wc. The scalar normalization must remain explicit. If the three vectors are mutually orthogonal, their norms are constant across corners and normalization creates no mixed term. Nonorthogonal vectors can generate one even though the raw mixed coefficient is zero.

The [executable CPU check](../rms_regeneration_gram_v1.py) verifies this identity and folded readers across 32 random fixtures, including residual width 1152, plus planted orthogonal and nonorthogonal controls. Maximum coefficient discrepancy is 1.17e-15; folded-reader discrepancy is 4.33e-15. This establishes a candidate transparent interface, not its discovery in the native model. The vectors are still context dependent and require native counterfactual inputs. The next native check can compare this Gram interface with the freshly measured raw-removal producer; it must preserve the failed structural cases.

This helps specify and potentially extract an operation while retaining its nonlinear dependence. It does not yet establish OOD semantic prediction, selective reuse across tasks, or an independently generated residual state. All 545,902,902 native parameters remain charged; no structural saving is claimed.

## 21. The preceding bilinear MLP contributes through both multiplication branches

The [07:49 mathematical review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0749.md) changes the next step. The Gram representation follows from general properties of normalization; another native replay of that identity would mainly validate execution. We instead tested whether MLP8 supplies the mixed value consumed by attention9.

Write the actual MLP8 left and right projected inputs as four conditional components: mean, object-number, attractor-kind, and their interaction. The mixed output is exactly

\[
D(L_oR_h+L_hR_o)+D(L_0R_{oh}+L_{oh}R_0).
\]

Products inside parentheses are coordinatewise; D is the learned output projection. The first branch multiplies separate signals to create an interaction at this MLP. The second processes an interaction already present in its normalized input. These are distinct parts inside one native module. Their names do not establish semantic interpretation of the input vectors.

We removed the whole mixed MLP8 write, the first branch, or the second branch from the raw input used by the partial attention9 value producer. Each removal includes the native residual scaling. RMS normalization and the later model recompute; native routing and other consumers remain in the background. This is more informative about this path than the earlier final-residual accounting, which omitted its subsequent nonlinear response.

The [native result](../MATURE_VALUE_MLP8_ORIGIN_V1_RESULT.json) is valid. The weight-based MLP partition matches the observed mixed MLP write within 1.44e-5 relative error. Execution used 320 forwards over 5,120 sequence instances in 5.67 seconds. The unchanged native answer logits replay exactly.

The MLP8 path is material: its effect magnitude is about 44–56% of full partial-component removal in original layouts, and 46–101.2% in fronted layouts, across the two readout objects. **These are vector norm ratios, not percentages explained.** MLP8 does not reproduce the full partial-component effect within 10% in any group. Nor does either multiplication branch alone reproduce the MLP8 effect within 10% in any group.

Both branches are needed. Their effects sum to the complete MLP8 path effect within the registered tolerance in all 32 groups; the largest full-table composition error is 0.246%. This establishes approximate composition for these particular removals, not arbitrary interventions or reuse across tasks.

The [executed signed accounting](../MLP8_VALUE_BRANCH_ACCOUNTING_V1_RESULT.json) clarifies the structural difference. In original layouts, the new-product branch's projection onto the complete MLP8 margin effect is 25.6–48.5%; the inherited branch contributes 51.5–74.4%. In fronted layouts, the respective ranges are −17.3–125.1% and −25.2–117.3%. Negative values and values above 100% indicate cancellation; they cannot be read as positive fractions of a circuit. The small interaction remainder closes the accounting.

This gives a weight-defined, testable split of a material upstream contributor, but does not make MLP8 a complete circuit. Its other consumers remain untested here. A useful next native comparison is whether actual MLP8 branch removal, with every downstream consumer allowed to respond, agrees with the isolated value-path effect. Agreement would support that path as a dominant consumer; disagreement would require identifying the other consumers instead of treating the native MLP boundary as a single operation. No such successor run is registered yet. OOD prediction, independent inputs, selective reuse and structural savings remain unresolved.

## 22. MLP8 has important consumers beyond the isolated value path

We next removed the same three branches at the actual MLP8 output, letting every later computation respond. Unlike section 21, this changes subsequent queries, keys, other values, residual carry and later MLPs as well. The native model applies its residual coefficients; no later computation is clamped.

The [all-consumer test](../MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json) is valid: the native baseline replays exactly, the weight-partition checks pass, and the run uses 320 forwards over 5,120 sequence instances in 5.42 seconds. The isolated value path fails the 10% agreement test for every branch in every group. For the full MLP8 mixed write, disagreement is 52.6–79.1% in original layouts and 56.6–89.8% in fronted layouts across the two readout objects.

Neither the new-product branch nor the inherited branch alone reproduces the complete MLP8 effect. Their effects nevertheless compose within the registered tolerance in all 32 groups; maximum full-table error is 1.31%. Thus the two-branch decomposition survives the broader consumer test, while the claim that this value path explains MLP8's effect does not.

The [post-result geometry](../MLP8_CONSUMER_EFFECT_GEOMETRY_V1_RESULT.json) finds that the value-path effect generally points in the same direction as the complete MLP8 effect, but is smaller. For the full mixed write, its signed projection onto the total correct-margin effect is 21.2–47.2% in original layouts and 14.1–45.4% in fronted layouts. This supports a partial value role. It does not identify a separate complementary circuit.

Why not simply subtract the value-path effect from the total? Effects can depend on whether the other route is also changed. An executed control uses F(x,a)=x+a+2xa, where x represents a source-dependent bypass and a an attention response. Removing both from their native value 1 has effect 4. Changing either alone has effect 3. Subtracting one isolated effect from the total gives 1, which is not the other isolated effect 3; an interaction of −2 closes the accounting.

The next native question therefore needs four actual combinations: native/edited MLP8 source crossed with native/edited attention9 output. This would separate the immediate attention response from the residual and later routes bypassing that response, while measuring their interaction explicitly. It is a candidate experiment, not a completed finding. The [CPU control](../mlp8_consumer_effect_geometry_v1.py) has been executed; no successor GPU job is registered yet.

The four-property goal remains incomplete. These results identify a useful within-module split and show why its consumers must be traced beyond one value component. They do not yet supply an independent semantic input generator or a reusable extracted circuit, and all native parameters remain charged.

## 23. Attention9 and its bypass are separate contributors with little interaction

The four-combination native experiment is now complete. The source is either native MLP8 or MLP8 with its complete mixed output removed. Independently, attention9 writes either its native response or its response to that source edit. All later layers recompute. This preserves the source-dependent residual background while making the attention response an explicit intervention.

Let F00 be the native run, F11 the fully edited run, and F01/F10 the crossed cases. We measure

\[
E_{total}=F_{00}-F_{11},\quad
E_{attention}=F_{00}-F_{01},\quad
E_{bypass}=F_{00}-F_{10}.
\]

The interaction is the total minus the two isolated effects. Here “bypass” means routes that do not require attention9's changed write; it includes residual carry and later computations, rather than naming a single semantic module.

The [native test](../MLP8_ATTENTION9_FACTORIAL_V1_RESULT.json) passes all implementation checks. The native run, source-edited run and both identity clamps replay exactly. The shared first-value state remains unchanged. Execution used 384 forwards over 6,144 sequence instances in 5.60 seconds.

Neither attention nor bypass alone reproduces the total effect within 10% in any of the 32 groups. Their interaction is small in every group: at most 0.0642% of the total mixed effect and 0.336% on the full output tables. These maxima cover the correct margin and centered three-answer readouts, not the full vocabulary.

The [executed signed accounting](../SOURCE_ATTENTION_ROUTE_ACCOUNTING_V1_RESULT.json) puts the attention contribution's projection onto the total correct-margin effect at 17.2–37.4% in original layouts and 11.5–43.4% in fronted layouts. The bypass projections are 62.6–82.8% and 56.7–88.5%, respectively. A small signed interaction closes the accounting. This quantifies the two routes for this intervention; it does not make either an independently extracted circuit.

The next useful distinction is inside the bypass. Does the MLP write travel directly through residual additions to the output reader, or must later computations respond? With all subsequent writes held fixed, a source displacement D reaches the final residual as

\[
\Delta r_{final}=\left(\prod_{\ell=9}^{17}\lambda_{\ell,0}\right)D.
\]

The final RMS, output weights and softcap can still be evaluated exactly. Our [CPU control](../source_attention_route_accounting_v1.py) verifies this transport, including negative residual coefficients and fixed embedding injections, to 2.67e-15 at the nonlinear reader. A planted live nonlinear write makes the same prediction fail, demonstrating why the fixed-write assumption matters. Native direct-carry dominance has not yet been tested for this source/bypass comparison.
