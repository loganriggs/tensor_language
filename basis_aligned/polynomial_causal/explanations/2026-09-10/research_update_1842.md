# Research update — 10 September 2026, 18:42 UTC

This is the **new consolidated update**, continuing the previous requested report through **14:17 UTC**. It covers completed experiments through **18:13**, their subsequent audits, and the workflow repair through **18:26**. The next last-layer experiment is still in preparation; it has no result yet.

The main advance is that we now have two concrete, partly reusable interventions: one changes **which verb** the model favors, and another changes **the verb's grammatical form**. We also have explicit weight-derived formulas for how these changes are read by the last bilinear layer and the output layer. However, the interventions interact, affect controls, and still depend on contextual information supplied by the original model. **We have not extracted a circuit satisfying all four requested properties.**

The work has narrowed the problem: a shared grammatical direction is useful, but its effect depends on other state coordinates. Neither the direction alone nor a correction for output normalization explains the behavior. We now need to explain the computations producing those other coordinates.

The high-level sequence was:

- **Weight folding:** pursued both individual output-token vectors and shared structure among those vectors. Individual-token folds predicted some local effects well; cluster means alone discarded essential differences.
- **Attention splitting:** tested whether the two behavior branches could be separated by their value or two query/key factors. The proposed clean split between the two query/key halves failed.
- **Shared grammar variable:** built one direction from verb-form contrasts in the unembedding. Editing its values across layers transferred much of the grammatical behavior, including to fresh verbs, but failed a closer agreement control.
- **Mathematical review:** derived executable local response programs and counterexamples showing why a few selected readers cannot determine the full output change.
- **Lexical versus grammatical commands:** measured the two commands separately and together. They transfer substantial behavior, but they do not behave as independent, selective circuits.
- **Latest saved-state analysis:** showed that changed contextual coordinates are necessary to explain the grammatical command's lexical side effects. This motivates the pending last-layer producer test.

All results below concern this research thread. They are not an inventory of every concurrent experiment in the shared repository.

## What happened to the two unembedding paths?

The **unembedding** is the weight matrix converting the model's final state into token scores. Each row is a **reader**: a vector that asks the final state a particular linear question, one per output token. Folding a reader backward means algebraically composing it with earlier weights to expose which earlier quantities it reads.

For a bilinear layer with input $u$, the computation is

$$
m(u)=D[(Lu)\odot(Ru)]+b.
$$

Here $L$ and $R$ produce two lists of features, $\odot$ multiplies matching entries, and $D$ combines those products into the output. A reader $w$ therefore reads

$$
w^\top m(u)=(w^\top D)[(Lu)\odot(Ru)]+w^\top b.
$$

The vector $w^\top D$ gives explicit coefficients on the layer's feature products. This is an exact weight identity. It does not, by itself, show that a small subset of those products is a reusable circuit.

**Individual-token path.** Folding token readers through the last two bilinear layers, while retaining live intervening attention, predicted the tested full-vocabulary intervention effects with roughly **4.6–5.6% relative error**. The associated cross-entropy prediction error was below **0.0034 nats**. Cross-entropy measures how much probability the model assigns to the correct token; lower is better. These are local predictions using native weights and context, not an independently extracted model. The edited earlier-layer contribution also slightly opposed the desired task effect, so accurate prediction was not evidence of a helpful task-specific branch.

**Structured path.** A hierarchy with 16 token groups was folded backward too. Keeping only group means left approximately **94–98% effect error**. The differences between individual token readers and their group means—the **token-specific remainders**—were essential. A separate comparison of 518 token functions found no qualifying proportional partners in the frozen sample. This rejects those particular whole-function sharing proposals; it leaves shared subcomputations within functions open.

A more useful structured object emerged from **contrasts**: subtracting the bare-form reader from the corresponding “-ing” reader and averaging over eight fixed training verbs. This gave the shared grammatical direction used in the later experiments.

Details: [token and hierarchy folds](unembedding_token_and_hierarchy_backward_folds.md), [token functions](token_readers_as_bilinear_functions.md), and [shared grammatical component](shared_gerund_component_from_unembedding.md).

## Did the attention weights reveal how to split a shared module?

We examined **OV**, the path that transforms values into attention outputs, and **QK1/QK2**, the two query/key score factors that this model multiplies to route information.

The weight audit found that the remaining branch still read the full 128-dimensional value-input space in **19 of 26 examined heads**, and 127 dimensions in the other seven. Thus, the proposed branch separation usually did not correspond to disjoint value-input spaces. Moreover, overlap within one head does not settle whether contributions cancel or combine across heads.

**Scope correction:** I interpreted the proposed split too literally. We have not tested whether the two behaviors use different input subspaces through the joint QK1×QK2 computation. Both may use both factors.

The subsequent intervention experiment did **not** support assigning one behavior branch to QK1 and the other to QK2. In its conditional comparisons, removing value information caused much larger losses, around **0.78–0.89**, than removing either individual query/key half, around **0.013–0.066**. The interaction terms remained substantial. These measurements do not make routing irrelevant: the tested values already contain effects of upstream computation.

Later, replacing only the shared first-layer value at the lexical cue transferred about **84% and 81%** of the lexical effect across two sentence frames. That is evidence for a useful cross-layer information path. Its agreement-control failure prevents treating it as a selective lexical circuit.

Details: [OV input readers](attention_ov_input_reader_overlap.md) and [QK/value dependencies](attention_qk1_qk2_value_dependencies.md).

## How much did the shared grammatical direction achieve?

Call the unit direction $e$. A **scalar coordinate** is the single number $e^\top h$ measuring a state along that direction. A **scalar write intervention** replaces that number in a module's output while initially preserving its other coordinates. Later layers remain free to respond.

At the final state alone, transferring this scalar recovered about **59–62%** of the grammatical cue's effect. Transferring only the last bilinear layer's output scalar recovered just **7–10%**. Much of the useful scalar had already arrived from earlier computation.

Installing donor scalars at all **36 attention/MLP outputs** produced much stronger transfer. On the fresh verb and cue panels, recovery was **95% and 86%**. Recovery compares the intervention's task-margin change with the native cue change; it is not a percentage of examples answered correctly.

The closer agreement control changed the conclusion about selectivity. Removing the scalar at both tested endpoints changed the correct-token cross-entropy by **0.548 nats on average in absolute value**. The earlier, narrower control had looked favorable; both results remain in the record.

Clamping this direction at the inputs of all 18 bilinear layers reduced target recovery by about **29 percentage points** in each frame. Thus, bilinear layers do consume the direction. But preserving only the scalar trajectory did not predict the full output distribution: other state coordinates changed materially.

Primary discussion: [scalar writes and live feedback](gerund_scalar_writes_and_live_feedback.md).

## What did the mathematical cycle actually contribute?

The scheduled **16:49 mathematical review** produced an executable local response rule. If the normalized input of a fixed bilinear layer changes from $u$ to $u+\delta e$, then

$$
m(u+\delta e)-m(u)=\delta F_eu+\delta^2a,
$$

where

$$
F_e=D\operatorname{diag}(Re)L+D\operatorname{diag}(Le)R,
\qquad a=D[(Le)\odot(Re)].
$$

The first term depends on the existing context $u$; the second is fixed by the weights and direction. This explains why a shared direction can be reused while its effect still depends on the sentence.

For two selected output readers, we compiled a small conditional response program and verified its algebra to roughly **10⁻¹³** in double precision. But native-context reuse tests failed the broader claim: even an exact prediction for those two readers left approximately **99–101% error** when used as a replacement for the full-vocabulary response.

The review also constructed pairs of synthetic inputs with equal selected reader states and equal norm, but different full output responses. These are mathematical counterexamples to the proposed state being sufficient. They are not evidence that those synthetic states occur naturally in text.

A later program included the actual output normalization and score cap. Along a local scalar edit, the token numerator is quadratic in $\delta$, and the squared norm is quartic. For two tokens, **11 context-dependent coefficients** predicted the tested local scores within roughly **8×10⁻⁶**. That program still requires initialization from the original model's contextual state.

**The math helped identify an exact conditional computation and falsify overly small state descriptions. It did not supply an independent producer for that state.**

Sources: [16:49 mathematical review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md), [native response test](../../NATIVE_RESPONSE_GATE_V1_RESULT.json), [token/norm program](../../GERUND_READOUT_FACTORIAL_V2_RESULT.json).

## The latest experiment: can lexical identity and grammatical form compose?

At **18:03**, we tested a two-by-two grid: original versus alternate verb, and bare-form versus “-ing” cue. Each example had four output-token readers, covering both verbs in both forms.

The **lexical command L** replaced the shared first-layer value at the lexical cue. The **form command F** installed the same set of grammatical scalars across all 36 outputs. Crucially, F was reused across both verbs without a verb-specific adjustment.

| Measurement | Frame 1 | Frame 2 |
|---|---:|---:|
| L: lexical recovery under bare / “-ing” cues | 84% / 84% | 81% / 79% |
| F: form recovery for original / alternate verb | 95% / 95% | 86% / 86% |
| F: unwanted lexical drift, relative metric | 11.5–11.7% | 11.9–16.9% |
| Joint command: correct choice among four tokens | 14/16 | 13/16 |

The registered lexical-preservation limit was **10%**, so F failed selectivity. L's form-preservation point estimates passed, but its full registered claim failed other conditions. The original model itself failed two separate “-ing” endpoint comparisons in frame 2; those examples were retained.

The agreement control now measured both relevant token scores, resolving an earlier limitation where only correct-token probability was available. L changed their relative margin by **0.233 on average in absolute value**, above the **0.10** preservation limit. Thus, there was a measured change in relative grammatical preference as well as the older probability failure.

Adding the separately measured L and F effects predicted the **actual joint intervention** fairly well on the selected four scores: approximately **4–5% relative error**. This is partial compositional evidence. It does not repair the joint command's wrong choices or show that it reproduces the native joint target.

Receipt: [lexical/form experiment](../../LEXICAL_FORM_INTERCHANGE_V1_RESULT.json).

## Why did the final CPU analysis matter?

We decomposed the final state into the shared scalar and its perpendicular remainder:

$$
h=se+h_\perp.
$$

For token reader $U_t$, define $a_t=U_te$ and $c_t=U_th_\perp$. The actual output score is

$$
z_t=30\tanh\left(\frac{a_ts+c_t}{30\rho}\right),
\qquad \rho=\sqrt{\operatorname{mean}(h^2)+\epsilon}.
$$

Here $\rho$ is the final root-mean-square normalization factor; the hyperbolic tangent caps extreme scores smoothly. This separates three possible sources of change: the shared scalar, the other coordinates' token contributions, and normalization.

At **18:13**, saved-state CPU tests checked whether simpler combinations explained F's unwanted lexical drift. Scalar-only prediction left **83–95% relative error**. Giving it the actual changed norm still left **82–107%**. The complementary contribution plus the changed norm also failed. The full formula reproduced the native scores within **6.3×10⁻⁶**.

Further accounting showed that scalar and complementary lexical contributions partly oppose each other. Their vector cosines were about **−0.30 to −0.62**: they partly cancel. This accounting uses both measured endpoints; it is an explanation of an observed change, not an independent prediction of an unseen intervention.

The four token readers were also decomposed exactly into a common component, a lexical contrast, a form contrast, and their interaction. The interaction coordinate accounted for **58–81% of squared lexical drift** under F. This locates drift in the output description; it does not establish which upstream computation caused it.

Receipts: [readout factors](../../LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json) and [cancellation accounting](../../LEXICAL_FORM_NUMERATOR_CANCELLATION_V1_RESULT.json).

## Where this leaves the four requested properties

| Property | Progress in this interval | Remaining gap |
|---|---|---|
| Prediction on unseen and out-of-distribution inputs | Fresh verbs/cues transfer; explicit local response formulas work under their stated context | No independently generated state or broad distribution-shift guarantee |
| Extraction | Explicit token readers, product coefficients and small conditional readout programs | Native context and substantial original computation remain necessary |
| Selective removal/manipulation | Strong, measurable grammar and lexical interventions | Closer controls and lexical/form cross-effects fail selectivity |
| Composition and reuse | Same form command transfers across verbs; separate effects approximate some joint scores | Joint choices fail, interactions remain, and full independent composition is unproved |

The next prepared experiment separates the **last bilinear layer's complementary output** from the **complementary state carried into that layer**. It will fold the four structured readers onto actual bilinear products and test which source accounts for the observed lexical drift. As of this report, only the executor extension and experiment preparation are underway: **no successor GPU result exists**.

## Execution and reporting

The 18:14 hourly review counted four owned native runs in the preceding hour: **167 model forwards, 2,672 sequences, and about 7.87 seconds inside their measured executors**. That last number excludes loading, queueing, authoring and analysis. Median spacing between native receipts was about **15 minutes**, slower than the 10-minute operating target. Publication occupied about **16.6 minutes** of the audited hour.

One earlier run failed while publishing its artifact because file size was mistakenly called as a function. Its evidence was preserved, the publisher was repaired, and the replacement run reproduced the saved states exactly. The repair is not counted as a scientific discovery.

To reduce repeated restoration and publication work, the startup guide was shortened from **4,335 to 215 lines**, with its full history preserved byte-for-byte. This report's delivery also took too long: preparation of the next experiment and repeated receipt checks delayed the requested explanation. The report is now a separate file, linked first in both explanation indexes.

Source: [18:14 hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-10_1814.md). The next scheduled reviews are **19:14 UTC hourly** and **19:49 UTC mathematical**, subject to newer live review records.
