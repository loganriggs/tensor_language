# Research update: shared producers and interaction-path decomposition

**12 September 2026, findings through 17:30 UTC. Requested full update for Logan.**

This covers work since the previous major report's **16:20 UTC addendum**, rather than counting its earlier results again. The previous file was [Interaction-path decomposition and extraction](research_update_2026-09-12_1536_interaction_decomposition.md). Layer/head numbering is zero-based: head8.2 is head 2 in block 8; block 17 is the last block. This report describes Codex's work, not Claude's separate experiments.

## High-level overview

**The strongest new result is a concrete grouping across two attention heads: distinct upstream computations contribute almost the same downstream variable, and that grouping survives fresh behavioral tests.** We now have an executable pair of producers feeding the previously extracted regional-spelling branch. This is meaningful progress toward decomposing interaction paths. It remains conditional on substantial native computation, and does not yet give circuits that simply fall out of a general unsupervised factorization.

The main events were:

1. **The promising head13.0 factor failed behavioral validation.** Its large share of coefficient energy did not translate into a large or faithful regional-spelling effect. We investigated a real coordinate-dependence problem in the objective, corrected it, and reran the comparison. The behavioral failure survived.
2. **An exact interaction decomposition showed where the effect lives.** Most measured transfer comes from products mixing upstream producers with the existing background. Keeping only terms linear in producer contributions loses 12–14% of the effect. However, the specific hypothesis of strong interaction between head13.0 and the other producers failed.
3. **A fixed bank of 27 weight-derived components found two useful producers: head8.2 and head9.8.** We reported all 27 results, including the 25 misses. Their writers are almost identical in the downstream four-reading space, while their source readings differ.
4. **The pair passed fresh lexical/template confirmation.** On 48 new prompts, each component closely reproduces its corresponding head's conditional effect. Together they retain 89–93% of the selected producer group's transfer. Merging their output directions changes the joint effect by only 0.23–0.25%.
5. **We compiled and tested the pair as an executable program.** It computes both complete QK1 × QK2 routing operations and scalar values; attention weights are not supplied as unexplained inputs. It still needs the original model's contextual states at layers 8 and 9.
6. **Selectivity remains unresolved.** A dedicated newline test failed its capability and whole-head positive controls. Tiny changes from the regional components therefore do not establish newline preservation. A follow-up comparing natural FineWeb contexts and two intervention types has frozen rows and a preregistration, but has not run.
7. **The latest math cycle identified an important upstream dependency: normalization.** Linear readings alone cannot generally generate these producers from raw residual states. An explicit norm observable is needed. A small additional weight calculation now also gives a principled way to remove each component at its actual producer, which is stronger than the consumer-edge interventions tested so far.

The large percentage needs its denominator: **89–93% is the share of a selected group's conditional transfer, not the share of the model's entire behavior.** Removing the pair reduces the full native regional cue contrast by approximately **7.6–8.5%** on this fresh panel.

## Status against your four properties

| Property | New evidence | What remains missing |
|---|---|---|
| OOD prediction | Frozen factors and their common writer transfer to new cities, spelling pairs, and prompt templates. | Broad natural-text and corpus-shift prediction; these controlled prompts are not proof of pretraining disjointness. |
| Extraction | A 5.54 MB producer program reproduces native contributions and the registered conditional interventions. | Independent generation of its layer8/9 inputs and closure of the downstream query, normalization, and background dependencies. |
| Selective removal | Consumer-specific removal consistently weakens the regional cue effect; an unrelated token-margin control is smaller. | Actual recursive component removal and a valid test preserving head8.2's known newline service. |
| Composition / reuse | Two distinct producers can use one downstream writer; joint effects were tested, not inferred from individual effects. | General reuse across tasks and a closed composed program whose intervening states are generated internally. |

No new component is being promoted as a completed four-property circuit.

## 1. What is being decomposed now?

The original proposal concerns folding the unembedding backward through bilinear and attention paths. The earlier report covered the joint full-unembedding path fits and their limitations. **There has not been a new successful global full-unembedding factorization in this interval.** The current work follows one concrete branch found along that route and decomposes its upstream producers.

That regional branch reads four numbers from a source state. Its polynomial numerator has the form

$$
F(q,f)=\beta_0(q)f_0f_1f_2+\beta_1(q)f_0f_1f_3.
$$

Here $f=(f_0,f_1,f_2,f_3)$ contains four linear readings, and $q$ is a query state. The vector-valued $\beta_j(q)$ specifies how each product writes downstream. The shared intermediate is $f_0f_1$; the two children multiply it by $f_2$ or $f_3$. Actual execution also retains positional effects, native normalization, and the later model computation.

We asked which earlier attention computations supply these four readings. Folding a consumer's readers into a producer's output/value weights gives

$$
M_i=s_i C O_i
\begin{bmatrix}(1-\mu_i)V_i&\mu_i V_{0,i}\end{bmatrix}.
$$

$C$ contains the four downstream readers; $O_i$ and $V_i$ are the head's output and current-value matrices; $V_{0,i}$ is its shared first-layer value map; $\mu_i$ is the learned signed mixture coefficient. The residual propagation factor $s_i$ includes the actual intervening residual re-entry coefficients. Thus $M_i$ maps a concatenated current/first source state into four downstream contributions.

The crucial distinction is that **we keep the full joint QK1 × QK2 computation while simplifying the value-to-consumer map**. We are not dividing the two tasks between QK1 and QK2. Both routing factors participate together in each producer.

The layers 8, 9, and 13 were selected using earlier behavioral tracing. The subsequent component construction uses weights without fitting text outcomes. This is therefore **behavior-informed choice of a path, followed by weight-based component discovery**, not a fully unsupervised search over the whole model. [Primary derivation and results](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md).

## 2. The head13.0 result failed, and its audit was informative

The previous report ended with a structured approximation retaining roughly 63–65% of head13.0's folded coefficient energy. Its native test completed after that cutoff.

The chosen value component supplied only **1.3–2.5% of the selected producer group's transfer**. Its effect differed from the full head by **59–141%**, missing the 10% fidelity bar. Most signs were correct, but the magnitude was inadequate. This is a direct example of a large weight-space component being a poor explanation of the particular behavior. [Original behavioral result](../../STRUCTURED_PRODUCER_EFFECT_CPU_V1_RESULT.json).

We then tested whether the objective itself was misleading. The four intermediate readings admit a coordinate change: scale one reading and compensate the downstream polynomial coefficients. The complete computation stays the same. An ordinary singular-value decomposition of $M_i$, however, can choose a very different leading direction after this harmless re-encoding. On the actual component, a 100-fold rescaling left the branch unchanged while the leading source-direction cosine could fall to **0.00577**. Apparent retained energy could exceed **99.8%**. This is a demonstrated coordinate confound, not speculation. [Gauge audit](../../FOLDED_PRODUCER_READER_GAUGE_V1_RESULT.json).

### The corrected weight-based objective

We defined a metric that measures changes in the four readings by their downstream polynomial effect:

$$
H=\mathbb E\left[J_fF(q,f)^T J_fF(q,f)\right],
$$

and solved

$$
\min_{\operatorname{rank}(\widehat M)\leq1}
\left\|H^{1/2}(M-\widehat M)\right\|_F^2.
$$

$J_fF$ is the derivative of the downstream output with respect to its four readings. A metric is the rule for measuring the size of an error. This one gives greater weight to reading changes to which the consumer is more sensitive.

The expectation uses independent isotropic Gaussian query/current states, a uniform sum over the complete vocabulary's first-state lookup, and 32 relative positions. Its polynomial moments are computed from weights. It does **not** use language examples to fit the component. Those distribution assumptions omit real contextual correlations, normalization gates, and final-suffix sensitivity; this is a principled search metric, not a behavioral guarantee.

For fixed $H$, a transformed matrix SVD solves the rank-one problem directly. There is no unresolved iterative optimizer convergence in this particular step. Under an invertible re-encoding $f'=Sf$,

$$
M'=SM,\qquad H'=S^{-T}HS^{-1},\qquad
M'^TH'M'=M^THM.
$$

Thus the selected source direction is invariant when its top eigenvalue is isolated. The construction passed its algebraic and sampling controls.

**It did not rescue head13.0.** The new source direction was very similar to the original one, and native fidelity errors remained **51–148%**, with only **1.6–3.0%** of group transfer. The correction improves the method; the negative behavioral result stands. [Metric controls](../../CONSUMER_PULLBACK_V1_CONTROL.json) · [Corrected behavioral test](../../CONSUMER_PULLBACK_EFFECT_CPU_V1_RESULT.json).

## 3. What the interaction decomposition actually found

We split the four readings into

$$
f=B+H_0+O,
$$

where $B$ is the recipient background, $H_0$ is head13.0's contribution, and $O$ is the remaining selected producer contribution. The subscript distinguishes this head contribution from the metric $H$ above.

Expanding each cubic product yields ten grouped interaction paths:

$$
B^3,\ B^2H_0,\ B^2O,\ BH_0^2,\ BH_0O,\ BO^2,
\ H_0^3,\ H_0^2O,\ H_0O^2,\ O^3.
$$

These symbols collect assignments to **different reader slots**. For example, $BH_0O$ includes all six ways to assign background, head13.0, and other producers to the three slots. It is not generally six times one scalar product, because the slots read different coordinates.

The decomposition is exact at the write level: summing its terms replays the original write to relative error about $10^{-7}$. We then sent interventions through the native suffix to measure token-margin effects.

| Comparison | Result | Interpretation |
|---|---|---|
| Seven mixed paths versus the full group swap | **0.16–0.29% effect error** | Almost all measured transfer comes from mixed terms under this partition. |
| Only terms linear in the producers, $B^2H_0+B^2O$ | **11.8–14.1% error** | Second-order producer terms matter; a purely linear treatment misses them. |
| Terms containing both head13.0 and other producers | **0.8–1.9% of full effect norm** | The proposed strong head13.0–other-producer cooperation did not appear. |

The largest additional term beyond the linear treatment is $BO^2$. This supports your interest in interactions, but the main signal here is **interaction with the background**, rather than a large interaction specifically between the initially favored head and other producers.

The exact expansion is a diagnostic. Ten written terms are not automatically a cheaper circuit than the original shared two-child expression. Background dominance can also arise when producer contributions are small relative to background. We have not shown that these ten paths are a universal sparse basis. [All path results](../../PRODUCER_INTERACTION_CUT_CPU_V1_RESULT.json).

## 4. Two different producers converge on one downstream variable

Using the same fixed consumer metric, we selected one value component for each of the 27 heads in layers 8, 9, and 13. Every candidate retained its complete QK operation. We did not tune ranks separately to make individual heads pass.

Only **head8.2 and head9.8** passed both the fixed conditional fidelity and transfer/control criteria on the reused 96-context screen. The other 25 results are retained. This bank is an exploratory screen with multiple candidates, so fresh confirmation was necessary. [All 27 outcomes](../../STRUCTURED_PRODUCER_BANK_EFFECT_V1_RESULT.json).

Their downstream writers have cosine **0.99949** in the consumer metric. Cosine near one means almost the same direction, after accounting for sign convention. Their combined current/first source readers have absolute cosine only **0.516**. The current sectors have absolute cosine **0.631**; the first-state sectors only **0.011**. Their input states also occur at different layers.

This is close to the distinction you wanted us to look for: **different input computations can share a downstream output**. It does not establish disjoint input spaces, nor that their full QK product spaces coincide. We have preserved those full routing maps rather than claiming they have already been split into minimal task-specific subspaces. [Geometry](../../STRUCTURED_PRODUCER_PASSERS_GEOMETRY_V1_RESULT.json).

### Fresh confirmation

We froze the two components and a common output direction before evaluating 48 new prompts: two new templates, four new cities, and six new spelling pairs. These include Bristol/Boston and Oxford/Austin, with pairs such as centre/center and honour/honor. Both cue directions are included, giving 24 paired cue contrasts rather than 48 independent concepts.

The original model showed the intended cue effect on all 24 pairs. The frozen components then gave:

| Measurement | Template family 1 | Template family 2 |
|---|---:|---:|
| Head8.2 component versus its whole-head conditional effect | 2.98% error | 2.79% error |
| Head9.8 component versus its whole-head conditional effect | 2.34% error | 2.98% error |
| Pair's transfer as a fraction of selected group transfer | 92.53% | 89.18% |
| Shared-writer join versus original pair's joint effect | 0.227% error | 0.245% error |
| Sum of individual effects versus directly evaluated joint effect | 2.47% error | 1.62% error |
| Reduction of full native cue contrast after pair removal | 0.263 | 0.152 |

The last row is in token-score margin units. Dividing by the native cue contrasts, 3.088 and 1.991, gives approximately **8.5% and 7.6%**. This is why the producer-group percentage must not be presented as whole-behavior coverage.

All 48 directed component/joint swaps moved the target margin in the expected direction. The selected unrelated token-margin control was substantially smaller. These are useful fresh conditional tests, with no confirmation-data refit. [Fresh capability and effect results](../../PRODUCER_FRESH_CONFIRMATION_EFFECT_V1_RESULT.json).

## 5. What the executable extraction computes—and still depends on

For each producer, the program computes

$$
a_i(t)=\sum_{s\leq t}\gamma_i(t,s)
\left[v_{i,\mathrm{cur}}^Tx_i(s)+\tau_i(\mathrm{token}_s)\right],
$$

where $x_i$ is that layer's native normalized contextual input, $\tau_i$ is a token lookup compiled from the actual first-state initialization, and

$$
\gamma_i(t,s)=
\frac{\langle\widetilde q_{1,i}(t),\widetilde k_{1,i}(s)\rangle}{128}
\frac{\langle\widetilde q_{2,i}(t),\widetilde k_{2,i}(s)\rangle}{128}.
$$

The tildes include native projected Q/K normalization and position rotation. There is no attention softmax in this model. The two scalar producers write through the shared four-coordinate vector $u$:

$$
\Delta f(t)=u\left[\alpha_8a_8(t)+\alpha_9a_9(t)\right].
$$

The $\alpha_i$ include the folded propagation scales and writer amplitudes. At the normalized downstream reader interface, contribution changes are divided by the recipient's actual residual17 RMS.

The package stores **1,282,566 scalars, or 5,541,936 tensor bytes**. Of those scalars, **1,179,648 are still complete QK maps**. Sharing the four-dimensional writer saves only two output scalars locally; the useful result is the common computational interface, not a large compression gain.

Native contribution replay errors are at most $3.02\times10^{-7}$ per producer and $8.75\times10^{-8}$ jointly. Compiled swap/removal effects agree with the frozen shared-writer reference within $6.86\times10^{-5}$. This numerical execution error is separate from the approximately 0.24% behavioral approximation introduced by merging writers.

**The unresolved extraction boundary:** the program does not generate layer8/9 contextual states from tokens alone. The downstream query, key normalization, other source contributions, and suffix also remain native. It is an executable conditional component, not an independently operating language circuit. [Package and usage](../../extracted_circuits/regional_shared_producers_8_2_9_8_v1/README.md) · [Native replay](../../SCALAR_PRODUCERS_NATIVE_V1_RESULT.json) · [Intervention replay](../../SCALAR_PRODUCERS_EFFECT_REPLAY_V1_RESULT.json).

## 6. The newline control is inconclusive, not a preservation success

The existing dossier already identifies head8.2 as part of a newline-setting group. We checked that prior work instead of treating its regional contribution as a newly discovered whole-head identity.

Our first dedicated control used 16 authored list/form prompts. It measured full-vocabulary newline cross-entropy, $-\log p(\text{newline})$, and compared conditional regional-edge removals with actual whole-head8.2 removal propagated through subsequent layers.

The numerical implementation checks passed, but the original model did not satisfy the registered newline-capability thresholds:

- Mean newline cross-entropy was **9.01 and 10.86**, exceeding the required ceiling of 5.
- Mean newline-minus-comma margins were **−6.57 and −9.16**.
- Whole-head zero removal **improved** newline cross-entropy by approximately **0.62 and 0.81**, rather than providing the registered damaging positive control.

The candidate regional-edge removals changed newline cross-entropy very little, but that does not establish preservation of a demonstrated service. The report records the overall preservation inference as **inconclusive**. [Completed V1 result](../../SCALAR_PRODUCERS_NEWLINE_V1_RESULT.json).

The audit found two differences from the older dossier experiment: it used **natural FineWeb next-newline targets**, and it replaced a head's output with its **average output**, rather than zero. Zero and mean replacement are different counterfactuals.

The prepared V2 therefore crosses both context types and both whole-head intervention types. It retains the original 16 authored prompts and adds 32 natural prefixes whose actual next token is newline, chosen without filtering on model scores. A separate 24-row cache partition supplies the control mean. The factors remain frozen. FineWeb is training-domain data; this is neither a new-corpus OOD test nor a verified document-disjoint split.

**At this report's cutoff, V2 has its preregistration and generated rows; its runner is not yet implemented and no V2 model result exists.** The planned price is 150 body forwards under a 300-second cap. [V2 design](../../SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_PREREGISTRATION.md).

## 7. Did the mathematical work help this interval?

Yes. The coordinate-invariant metric and exact interaction expansion above were immediately implemented and tested. The scheduled 17:06 mathematical review added a separate result about how far backward we can fold the input interface.

Stack a producer's four 128-dimensional Q/K maps and its one current-value reader into

$$
E=\begin{bmatrix}Q_1\\K_1\\Q_2\\K_2\\v_{\mathrm{cur}}^T\end{bmatrix}
\in\mathbb R^{513\times1152}.
$$

Both producers have numerical row rank 513. For an already normalized input, projecting into the row span of $E$ preserves all those readings and the computed scalar producer. That replay passed near machine precision.

For a **raw residual** $r$, however, the computation first uses

$$
x=\frac{r}{\rho(r)},\qquad
\rho(r)^2=\frac{\|r\|_2^2}{1152}+\varepsilon.
$$

A perturbation $\delta$ can satisfy $E\delta=0$ while changing $\rho(r+\delta)$. Thus the raw linear readings stay fixed but the normalized computation changes. In the executed control, this changed producer outputs by **15.4% and 24.2%**. Adding the actual scalar $\rho^2$ repaired the calculation to near machine precision, and within approximately $4.2\times10^{-7}$ of native FP32 execution.

This gives a concrete requirement for deeper folding: **generate the needed readings and norm observables together**. Norm evolution itself includes cross terms:

$$
\|r+G(r)\|^2=\|r\|^2+2r^TG(r)+\|G(r)\|^2.
$$

When $G$ is quadratic, these terms have degrees two, three, and four. A proposed small state must preserve those dependencies or explicitly measure their approximation error. A dense reader encoder also increases this implementation's map storage from 590,976 to 854,145 scalars per head, so it was not adopted as a cheaper program.

The review mapped exact observable reduction methods from polynomial/rational systems to this problem and identified the assumption mismatch with native RMS and tanh. It supplies a useful closure test, not a theorem that the full transformer has a recoverable small polynomial representation. [Review and literature mapping](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_1706.md) · [Executed norm control](../../SCALAR_PRODUCER_INPUT_QUOTIENT_V1_RESULT.json).

### A further calculation completed while preparing this update

The tests above remove a contribution only at the regional consumer's four-reading interface. To test actual recursive producer removal, we need its physical residual write. For each original, unmerged source reader $v_i$, define

$$
d_i=O_i\begin{bmatrix}(1-\mu_i)V_i&\mu_iV_{0,i}\end{bmatrix}v_i.
$$

Then $d_i a_i(t)$ is the corresponding native-space value-component write, and its direct downstream projection satisfies

$$
s_i C d_i=w_i,
$$

where $w_i$ is the previously frozen four-coordinate writer. The new CPU calculation verifies this identity to relative error below $4.3\times10^{-16}$ for both producers. Their physical residual writers have cosine **0.899**, compared with approximately 0.999 in the downstream read space. They are more similar **as seen by this consumer** than as complete residual writes.

This provides a natural intervention for the next stronger test: subtract each component where its producer writes it, then recompute all later layers. No arbitrary pseudoinverse lift is needed. **That intervention has not yet been run.** It may affect other consumers and normalization, so the present conditional success does not predict automatic preservation. [New physical-lift receipt](../../SCALAR_PRODUCER_NATIVE_LIFT_V1_RESULT.json).

## 8. What this changes about the project direction

The interaction-path approach is now producing a more useful object than an isolated tensor approximation: **two explicit upstream arithmetic computations, a shared downstream variable, and a consumer with tested joint effects**. This is the strongest advance since the last report.

It also sharpens the remaining gaps. We have not found a generally successful sparse decomposition of the whole folded model. We have not exhausted richer shared QK subspaces or joint DAG fitting. The current positive result depends on a behavior-informed path choice, and its broader selectivity and independently generated inputs remain open.

The immediate priorities are to finish the registered natural newline control, evaluate physical component removal with recursive execution, and use the norm-aware interface when attempting further backward closure. A failed global removal would be evidence about the mismatch between consumer-specific grouping and a whole-model unit—not grounds to discard the conditional result or declare structure absent.

At the **17:28 UTC operational check**, both managed runners were running, lane1's queue was empty, and the latest logged task had completed. No decomposition experiment was then running; the newline follow-up was preparation, not a background job. The mathematical and report work in this interval ran on CPU. That distinction matters when interpreting how much computation has actually happened.

For detailed receipts, start with [the producer primary note](../../FOLDED_PRODUCER_NATIVE_V1_MATH.md), [the fresh confirmation](../../PRODUCER_FRESH_CONFIRMATION_EFFECT_V1_RESULT.json), and [the executable package](../../extracted_circuits/regional_shared_producers_8_2_9_8_v1/README.md). The [original proposal](interaction_path_decomposition_proposal_2026-09-11.md) and [previous report](research_update_2026-09-12_1536_interaction_decomposition.md) retain the broader full-unembedding, sparse-path, and hierarchy context.
