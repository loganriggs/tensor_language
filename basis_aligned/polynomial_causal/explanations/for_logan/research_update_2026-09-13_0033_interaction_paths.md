# Research update: following the interaction path backward

**13 September 2026. Scientific results through 00:29 UTC.** This is the new full update after [the previous review](method_redteam_and_roi_2026-09-12.md), whose closing addition covered the fresh-context failure at 23:44. I briefly recap that failure so the subsequent work makes sense without reading the previous report.

## 1. High-level overview

**The main progress is a more specific, causally tested interaction path across several modules.** We started with four weight-derived readings inside MLP8 that contribute to a selected value channel of attention head 9.8. Folding those readings backward exposed interactions between MLP7, the residual stream and attention8. Interventions then traced most of one attention-dependent contribution to the already known head 8.2.

This is evidence for your idea that **interactions across module boundaries can be better explanatory units than entire modules or independently decomposed layers**. In this case, treating the upstream contributions independently discards large mixed terms. Keeping those terms lets us distinguish a computation that participates in an effect from the input that supplies its changing information.

There is an important qualification: the branch does not consistently push toward the expected British/American spelling. In one new sentence construction, its overall effect reverses. We investigated that failure rather than averaging it away. The city-token source still pushes in the expected direction; later source positions push more strongly in the opposite direction. The newly traced interaction explains part of that opposition.

The sequence of substantive changes was:

1. **A fresh-context test failed.** Using the complete MLP8 quadratic instead of four modes did not remove the reversal.
2. **Splitting by source position exposed opposing contributions.** City-source donation had the expected sign on all 96 tested recipient prefixes; later sources reversed it in the problematic construction.
3. **Folding through MLP7 exposed essential mixed terms.** Removing the mixed terms from the algebra missed 58–71% of the city-cue contrast in the four-reading computation.
4. **Physical interventions localized the changing signal.** A path containing MLP7 was important, but changing the MLP7 readings alone explained little of the reversal. Much of the changing signal entered through residual and attention partners.
5. **Head 8.2 explained the attention partner contribution closely.** Its path-specific donor effect was within 5.54% of the full attention8 partner effect in the problematic construction.
6. **Two attempts to replace four readings with one failed.** The second solved its restricted weight-only approximation problem exactly. We retain four readings and prioritize tracing their inputs over further scalar compression.

We have a better conditional computation, not an independently extracted circuit with all four desired properties. The strongest new evidence concerns cross-boundary specification, targeted intervention and local composition. Broad generalization and autonomous extraction remain unfinished.

## 2. What we measure, and what the percentages mean

The task uses paired prompts that differ in a British versus American city cue, followed by a spelling-sensitive continuation. Examples of endpoint pairs are *humour/humor*, *catalogue/catalog* and *organising/organizing*. The native model is the original, unmodified model.

For each prefix, we measure a **logit margin**: the model's score for the British continuation minus its score for the American continuation. The native paired contrast is the difference between that margin under the two city cues.

A **donation** takes a specified intermediate quantity from the paired donor prompt, inserts its contribution into the recipient computation, and reruns the remaining model. We orient the resulting margin change so that movement toward the donor's regional preference is positive. Reported transfer percentages divide the mean directed change by the mean native paired contrast.

Thus “−5.79% transfer” means an effect opposing the donor's expected regional direction, with magnitude about 5.79% of the native paired contrast. It is not an accuracy change, a percentage of explained variance, or a CE change in nats.

There are also two different error scales:

- **Native-contrast error:** how large an approximation error is compared with the whole regional contrast.
- **Branch-relative error:** how large it is compared with the particular branch's intervention effect.

Both matter. In the near-quote construction, four modes versus the complete quadratic differ by only 1.72% on the native-contrast scale, but by about 52% relative to the complete branch's own donor effect. A small branch should not receive an easy fidelity pass merely because the whole model's contrast is larger.

The new diagnostic sequence uses 96 prefixes: 24 original anchors and 72 initially fresh prefixes, comprising three constructions, two city pairs, six spelling endpoints and two cue directions. Once inspected, these 72 prefixes became diagnostic data. Repeated follow-ups on them are **not additional held-out confirmations**. The different suffix endpoints also do not make the earlier city-token hidden state an independent observation each time.

## 3. Why the initially promising branch failed in a new context

The preceding report described a four-mode MLP8 value path with roughly 12.32% donor transfer on the original panel. The fresh constructions placed the city cue in different syntactic contexts: fronted, close to a quotation, or farther away before additional text.

| Construction | Four-mode donor transfer | Complete-quadratic donor transfer |
|---|---:|---:|
| Fronted city context | +9.03% | +9.76% |
| City close to quotation | **−4.10%** | **−2.35%** |
| More distant city context | +10.65% | +11.26% |

All 36 fresh native paired contrasts had the expected regional sign. The near-quote failure is therefore not explained by the original model lacking the behavior there. Removing the broader selected head9 component still affected roughly 36% of native contrast in that construction.

The strongest immediate objection was that four eigenmodes discarded a necessary corrective term. The complete-quadratic control directly tested that explanation: it reduced the reversal's magnitude but retained its sign. This rules out that simple repair; it does not imply that the larger regional mechanism is absent.

We then separated contributions originating at the city token from contributions originating after it:

| Construction | City-source donation | Later-source donation | Full branch donation |
|---|---:|---:|---:|
| Original | +3.09% | +9.01% | +12.32% |
| Fronted | +2.78% | +6.13% | +9.03% |
| Near quote | **+4.68%** | **−8.68%** | **−4.10%** |
| Distant | +1.46% | +9.05% | +10.65% |

City-source donation had the expected direction on **96/96** recipient prefixes. Near-quote later-source donation had the opposing direction on **24/24**. The separately measured city and later effects approximately composed: their sum differed from the full intervention by 1.41–2.68% relative error across constructions.

This is a useful split, but “source” means a position from which head9 reads. It does not yet specify where that position's contextual value was generated. Also, injecting the change only at the final recipient position missed 22–45% of the full intervention effect. The earlier recipient positions and their downstream consequences matter.

A separate attempt to generate the city value directly from its token embedding got the signs right but missed magnitudes badly: about 679% relative error overall. Even an oracle choosing the best context-independent city-pair contrast had a 24.67% error lower bound across the fresh contexts. That oracle is a diagnostic bound, not a fitted deployed circuit. We need a contextual input generator, not just a city lookup table.

Evidence: [fresh-context and complete-quadratic analysis](../../MLP8_VALUE_FRESH_V1_MATH.md), [physical source split](../../PHI4_SOURCE_DONATION_V1_RESULT.json), [token-only check](../../PHI4_CITY_TOKEN_GENERATOR_V1_RESULT.json).

## 4. The four-mode computation, built from the weights

Here is the object we folded backward. The residual width is $d=1152$, and each bilinear MLP has product width $m=4608$. Write MLP8 as

$$
\operatorname{MLP}_8(x)=D_8\big[(L_8x)\odot(R_8x)\big]+b_8,
$$

where $L_8,R_8\in\mathbb R^{4608\times1152}$, $D_8\in\mathbb R^{1152\times4608}$, and $\odot$ multiplies corresponding coordinates.

Let $r\in\mathbb R^{1152}$ be the frozen downstream value reader of the selected head9.8 path. Reading the MLP output gives

$$
r^T\operatorname{MLP}_8(x)=x^TM_8x+r^Tb_8,
$$

$$
M_8=\operatorname{sym}\!\left(L_8^T\operatorname{diag}(D_8^Tr)R_8\right),
\qquad \operatorname{sym}(A)=\frac{A+A^T}{2}.
$$

Only the symmetric part matters because $x^TAx=x^T\operatorname{sym}(A)x$. This is an exact weight composition, with no activation fitting.

An eigendecomposition expresses the symmetric quadratic as weighted squared linear readings. Retaining the four eigenvalues of largest absolute magnitude gives

$$
\phi_4(x)=\sum_{i=1}^4\lambda_i(u_i^Tx)^2
=(U^Tx)^T\Lambda(U^Tx),
$$

where $U=[u_1,\ldots,u_4]\in\mathbb R^{1152\times4}$ and $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_4)$. The eigenvalues are approximately $-335.35,+253.78,-118.88,-91.14$.

A **mode** here means one linear reading $u_i^Tx$, its square, and its signed coefficient. It is not automatically a semantic feature or an independently identified circuit. These four modes were obtained from weights, but choosing the downstream reader was informed by the regional-behavior investigation. This is not a wholly unsupervised discovery of the entire model.

## 5. Folding backward exposes six interacting contributions

MLP8 receives a normalized contextual state. Let its pre-normalization state be $z_8$, and define the squared RMS denominator

$$
s_8=\frac{\|z_8\|^2}{1152}+\epsilon,
\qquad x_8=\frac{z_8}{\sqrt{s_8}}.
$$

Then

$$
\phi_4(x_8)=\frac{(U^Tz_8)^T\Lambda(U^Tz_8)}{s_8}.
$$

The actual block structure lets us split the four raw readings exactly:

$$
U^Tz_8=B+Q+H,\qquad B,Q,H\in\mathbb R^4.
$$

The three inputs are:

- $B$: readings of the incoming residual, learned initial-state re-entry, and the previous MLP's output bias.
- $Q$: readings of the previous bilinear MLP7's product contribution.
- $H$: readings of attention8's output.

More explicitly, with the actual learned block8 mixing coefficients $\alpha_8,\beta_8$,

$$
B=\alpha_8U^Tz_7+\beta_8U^Tx_0+\alpha_8U^Tb_7,
$$

$$
Q=\alpha_8 C_7\big[(L_7x_7)\odot(R_7x_7)\big],
\qquad C_7=U^TD_7\in\mathbb R^{4\times4608},
$$

$$
H=U^Ta_8.
$$

Here $z_7$ is the residual state before MLP7, $x_7=\operatorname{RMS}(z_7)$, $x_0$ is the model's initial normalized embedding state, and $a_8$ is attention8's output.

Substitution produces six terms:

$$
\phi_4=
\frac{
B^T\Lambda B+Q^T\Lambda Q+H^T\Lambda H
+2B^T\Lambda Q+2B^T\Lambda H+2Q^T\Lambda H
}{s_8}.
$$

This is the concrete connection to your interaction-path proposal. The mixed term $2Q^T\Lambda H$ combines information from two different upstream computations. It cannot be recovered by retaining only their separate squared effects.

The native replay verified the folded readings and value to roughly $10^{-6}$ relative error. Omitting all mixed terms missed **58–71%** of the city-cue contrast in $\phi_4$ across the four constructions. That number concerns the intermediate computation, not a claim that mixed terms explain 58–71% of final regional behavior.

Each coordinate of $Q$ is itself a quadratic form in $x_7$. Consequently, $Q^T\Lambda Q$ is quartic in that normalized input. However, the full computation includes normalization and contextual attention; it is not globally an ordinary degree-four polynomial in raw token embeddings.

Evidence and complete derivation: [MLP7/four-reader folding note](../../MLP7_PHI_READERS_FOLD_V1_MATH.md), [native parent replay](../../MLP7_PHI_PARENTS_V1_RESULT.json).

## 6. A path can contain MLP7 without receiving its changing signal from MLP7

We grouped all terms involving $Q$ together:

$$
F_Q=\frac{Q^T\Lambda Q+2Q^T\Lambda(B+H)}{s_8},
\qquad
F_{\mathrm{free}}=\frac{(B+H)^T\Lambda(B+H)}{s_8}.
$$

Their sum is exactly $\phi_4$. Donating the generated $F_Q$ value at later source positions produced **−10.75%** transfer in the near-quote construction, with opposing directions on 24/24 prefixes. Donating the $Q$-free group produced +2.15% on average, though only 16/24 directions were positive. Separately measured group effects composed within 0.41–2.53% relative error across the city/later partitions and constructions.

It would have been tempting to conclude that “MLP7 supplies the reversal.” But donating a composite value changes all its donor arguments, including $B$, $H$ and the denominator. To test the stronger interpretation, we first evaluated all eight donor/recipient combinations of $Q$, $B+H$ and $s_8$, then physically tested individual input swaps through the generated path.

| Input readings donated, with other inputs retained from recipient | Near-quote transfer | Error versus full $F_Q$ donation |
|---|---:|---:|
| All inputs | −10.75% | Reference |
| $Q$ only | −0.82% | 94.16% |
| $B$ only | −4.06% | 62.43% |
| $H$ only | −5.98% | 46.22% |
| $B+H$ | −9.97% | 17.33% |
| $Q+B+H$, recipient denominator | −10.77% | 0.63% |

**MLP7 participates in the operation, while much of the changing information in this context arrives through its partners.** With $Q$ and the denominator fixed, the partner change is exactly

$$
\Delta F_Q=\frac{2Q^T\Lambda\Delta B}{s_8}
+\frac{2Q^T\Lambda\Delta H}{s_8}.
$$

The separately measured $B$ and $H$ intervention effects predicted their joint effect within 0.35–1.00% across constructions. This checks composition after the remaining nonlinear model, rather than merely assuming that additive injected values imply additive final logits.

These are interventions on specified inputs of a generated path. They do not replace the entire MLP7 or attention8 output everywhere. Also, partner-only donation approximated the full path poorly in other constructions; we cannot generally discard $Q$ or normalization.

Evidence: [path-group donation](../../MLP7_PHI_PATH_DONATION_V1_RESULT.json), [input-allocation audit](../../MLP7_QPATH_INPUT_ALLOCATION_V1_RESULT.json), [physical input swaps](../../MLP7_QPATH_PORT_DONATION_V1_RESULT.json).

## 7. Folding attention8 identifies the head8.2 contribution

Attention8 has nine heads, each with a 128-dimensional value space. For head $h$, fold the four MLP8 readers into its output map:

$$
C_h=U^TO_{8,h}\in\mathbb R^{4\times128}.
$$

We can fold further through its current-stream and shared first-layer value maps:

$$
A_h=(1-\mu_8)C_hV_{8,h},
\qquad A_{0,h}=\mu_8C_hV_{0,h},
$$

both of shape $4\times1152$. The actual learned mixing parameter is $\mu_8$. With $\gamma_{8,h}(j,k)$ denoting the head's joint attention score from source $k$ to position $j$,

$$
H_j=\sum_h\sum_{k\le j}\gamma_{8,h}(j,k)
\left[A_h\widetilde x_{8,k}+A_{0,h}\widetilde x_{0,k}\right].
$$

The tildes distinguish the actual normalized attention inputs from the MLP inputs above. The first-layer attention input is not silently replaced by a raw token embedding. Each $\gamma$ retains the product of **both** normalized, position-rotated QK scores; this model does not use softmax attention.

This gives an explicit nested path:

$$
\text{source }k
\xrightarrow{\text{attention8}}
H_j
\xrightarrow{\;2Q_j^T\Lambda H_j/s_{8,j}\;}
\text{MLP8-derived value at }j
\xrightarrow{\text{head9.8}}
\text{recipient position }t.
$$

The original native head9 routing, value normalization and writer remain in the executor, followed by the native suffix. The construction exposes the internal computation without claiming those contextual dependencies have already been eliminated.

We checked existing dossiers before testing head8.2: it was a known regional producer, not a newly discovered head. The new question was whether it supplies this particular attention-partner input. The full head-reading cache replayed $H$ to about $3.6\times10^{-8}$ relative error.

**Physical head8.2-specific input donation then reproduced the full attention8 partner effect closely.** In the near-quote construction:

- Full attention8 partner donation: **−5.98%** transfer.
- Head8.2 partner donation: **−5.79%**, opposing on **24/24** prefixes.
- Relative effect error: **5.54%**.

Errors across all four constructions were 0.54–5.54%. Separately measured head8.2 and remaining-head effects composed within 0.022–0.103%. On the original panel the full attention-partner effect was tiny and directionally mixed; faithfully reproducing it is not evidence of a large signal there.

This supports a specific head8.2 → MLP8 interaction → head9.8 path. It does not say whole-head8.2 donation would have the same effect, nor that the four readings are four separate semantic circuits.

Evidence: [attention fold control](../../ATTENTION8_PHI_READER_FOLD_V1_CONTROL.json), [head cache](../../ATTENTION8_PHI_HEADS_V1_RESULT.json), [physical head8.2 confirmation](../../ATTENTION8_PHI_HEAD_DONATION_V1_RESULT.json).

## 8. Can one scalar replace the four readings? Two tests said no to their proposed approximations

A scalar representation would be attractive: one head-value reading multiplied by one MLP7-derived quadratic modulator. We tested two weight-only constructions, with a registered requirement of at most 10% error in the conditional donor write fields in every construction.

First we took a singular value decomposition of $|\Lambda|^{1/2}C_{8.2}$. SVD finds the optimal rank-one approximation to that matrix in Frobenius norm—the square root of the sum of squared entry errors. The single component captured 49.21% of its squared norm, but donor-field errors were **13.52–31.52%**. It failed.

The strongest objection was that this objective preserves head readings without accounting for their interaction with $Q$. We therefore incorporated the four MLP7 quadratic forms. If $Q_i(x)=x^TM_{7,i}x$, define their coefficient Gram matrix

$$
G_{ij}=\langle M_{7,i},M_{7,j}\rangle_F,
\qquad Z=G^{1/2}\Lambda C_{8.2}.
$$

An SVD of $Z$, with the left factor decoded through $G^{-1/2}$, solves the rank-one approximation of the **separated quadratic-input/head-value coefficient tensor** in its Frobenius metric. This is an equation-specific, weight-only method; it does not fit the prompt data.

Coefficient capture improved to 53.74%, but actual donor-field errors worsened to **44.91–97.84%**. Numerical controls passed and the Gram matrix was well conditioned. This is not an unconverged local optimizer: the restricted approximation problem is solved exactly, but its coefficient objective does not preserve these contextual effects well.

Counter-review matters here too. The failures do not establish that every possible scalar approximation on language must fail. They reject these two candidates. All four singular values of the composed object are nonzero, so its exact separation rank is four under the specified independent quadratic-input/linear-value partition. That is not a lower bound on every arithmetic rewrite or every approximate text-domain implementation.

**Decision: retain four readings.** Their tested interaction is already useful; tracing their inputs has higher explanatory value than forcing a one-scalar representation.

Evidence: [first scalar test](../../ATTENTION8_PHI_SCALAR_V1_RESULT.json), [composed-objective control](../../ATTENTION8_PHI_COMPOSED_SCALAR_V1_RESULT.json).

## 9. Did the mathematical work help?

Yes, in concrete ways.

**The normalization review clarified what an executable interface needs.** Four raw readings alone do not generally determine the normalized quadratic: a component outside their span can change the RMS denominator. Supplying four readings plus one norm scalar reproduced the tested native computation to about $1.3\times10^{-7}$ relative error. The mathematical counterexample establishes an unrestricted-input limitation, not that every constructed counterexample occurs on natural text. See [the 23:28 mathematical review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_2328.md).

**The six-term expansion supplied useful intervention units.** It told us what to keep together when testing an MLP7-containing path, and why separate module effects would miss interactions.

**The input decomposition prevented a false mechanistic conclusion.** A module's appearance in a polynomial does not prove it supplies the changing information. The individual-input tests materially changed the explanation.

**The composed Gram/SVD calculation gave a clean negative.** It answered the plausible objective-mismatch objection without a large optimizer campaign. Its limitation is now mathematical and empirical, rather than unresolved convergence.

Your normalized tensor-similarity proposal remains a deferred option for a situation where the candidate structure and contraction objective are a good match. We have not built the general structure-guessing optimization framework. The bespoke Gram/SVD control above was a small analytic calculation, not that deferred framework.

## 10. Red-team, counter-review, and the four desired properties

The strongest concerns are repeated use of a small controlled panel, borrowed native contextual inputs, externally annotated source positions, and selecting the downstream reader from an already studied behavior. Exact folding does not remove those dependencies. The newer path splits also do not automatically inherit the older component's preservation results; broad unrelated-behavior controls remain necessary for promotion.

The counterweight is that the useful results are not just tensor similarity or retrospective plots. We ran physical value-path and input-specific donations, reran the native suffix, retained the failed fresh-context criterion, and tested whether separately measured effects predict their joint effect. The decomposition explains a previously puzzling reversal without asserting that the reversal disappeared.

| Desired property | What improved | What is still missing |
|---|---|---|
| Held-out/OOD prediction | Fresh constructions exposed a real limitation and sharpened the mechanism. Earlier conditional interaction-prediction results remain in the prior report. | The broad positive four-mode claim failed; the newly refined path needs untouched contexts and broader text. |
| Extraction/sufficiency | Explicit four-reading equations and a conditional executor across module boundaries. | Native prefix states, QK routing, norms and suffix are still supplied. No autonomous small circuit. |
| Selective removal/manipulation | Source, path-group and individual-input donations localize signed effects. | Broad preservation and removal tests for the newly refined path; older control passes cannot simply be inherited. |
| Composition/reuse | Separate physical effects approximately predict joint effects; known head8.2 participates in a more specific downstream interaction. | General reuse across tasks and unseen conditions, and joint installation of independently extracted pieces. |

This is progress toward circuit interaction decomposition, but not yet general sparse-Tucker or DAG discovery over the full unembedding and every upstream path. We are following one behavior-relevant path with exact weight folding and causal tests. Its mixed-term structure is evidence that this unit of analysis is useful; it does not establish that the complete model admits a globally sparse interaction graph.

## 11. Costs, current status, and the next informative step

The GPU intervention bodies were inexpensive: the 672-forward path-group test took 9.59 seconds, the 768-forward input test 10.79 seconds, and the 384-forward head8.2 confirmation 5.77 seconds. These are measured experiment-body times, not end-to-end turnaround: loading, queueing, derivation, implementation, analysis and reporting add time. The composed scalar CPU calculation took about 0.52 seconds.

The bottleneck in this sequence is deciding and implementing the right intervention, not spending hours optimizing these small matrices. Ten random starts would not repair an SVD solution that is already optimal for its stated objective. For nonconvex structure searches, restart and convergence concerns still apply; they are simply not the failure mechanism of these two scalar checks.

Storage counts also need honest boundaries. The MLP7 reader-fold artifact has 23,050 scalars, but still references over 10.6 million native left/right MLP7 weights and contextual inputs. The attention-value fold has 87,553 scalars, with QK and contextual generation still external. Neither number is the size of a self-contained extracted model.

At the 00:33 status check, both managed runner services were running, lane1's queue was empty, and its latest recorded experiment had completed successfully at 00:23:05. That means the runner was available; it does not mean a new GPU experiment was underway. Disk space was about 1.4 GB free. This reporting step did not require deleting artifacts.

The next useful question is **what head8.2 reads to generate this particular partner signal**. We should split its current-stream versus shared first-layer values, retain the actual joint QK routing, and test those contributions at the already specified interaction edge. This reuses the head's dossier and the folded value maps. It can improve the computation's specification and eventual extraction without spending another round trying to collapse four readings into one.

The source-position and individual-input results suggest a practical route forward: follow a causally supported interaction backward, keep its contextual interfaces explicit, and test each proposed simplification before calling it an extracted circuit.

**Publication-time continuation:** A small weight-only check of the two head8.2 value maps is now complete. Their coefficient cosine is 0.296; the best proportional approximation leaves 95.5% relative residual. We therefore retain distinct current-stream and first-layer input ports for the next native test. Since the maps act on different contextual states, this does not determine their behavioral contributions. [CPU receipt](../../ATTENTION8_PHI_VALUE_SECTOR_PREFLIGHT_V1_RESULT.json).
