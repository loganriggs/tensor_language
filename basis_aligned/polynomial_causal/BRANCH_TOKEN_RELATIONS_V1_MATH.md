# Frozen branches, spelling relations and grammatical transfer

11 September 2026. The weight-derived two-branch node responds usefully to **subject–verb agreement and count-noun number**, but transfers only 3.6–4.5% of the complete model's answer change. Its registered 10% transfer criterion fails. Native last-MLP-only transfer is substantially larger, so this time the small effect cannot be explained by an absent positive local MLP signal. This is a partial component screen, not an identified or sufficient circuit.

## From token lists to shared output contrasts

Prior work already listed the top tokens, optimized output rotations, and measured how much the native layer reads the frozen input direction. Repeating those analyses would not identify a new computation. Instead, the [new atlas](BRANCH_TOKEN_RELATIONS_V1.json) evaluates all 25 frozen canonical branches from 12 shared parents against six explicit spelling transformations. These transformations are supplied as annotations; they were not discovered unsupervised. Factor discovery remains weight-only, and no text is used to fit them.

Each pair consists of two real, single GPT2 tokens with leading whitespace. The source word has at least three lowercase ASCII letters. The deterministic transformations and complete vocabulary-pair counts are:

| Transformation | Pairs |
|---|---:|
| Append `s` | 3,037 |
| Append `es` | 333 |
| Replace final `y` by `ies` | 184 |
| Append `ed` | 1,008 |
| Append `ing` | 921 |
| Capitalize initial letter | 6,262 |

These are spelling relations, not guaranteed grammatical inflections: the lists can contain noun plurals, verbs, fragments and accidental string matches. Complete pair lists and all branch scores are saved, including failed cells.

For a branch writer $w_j$ normalized to unit norm in the full-unembedding metric, let

$$
f_{tj}=(UW^{-1}w_j)_t,\qquad
\delta_{ij}=f_{b_i j}-f_{a_i j},
\qquad W^\top W=U^\top U.
$$

The pair $(a_i,b_i)$ is the original/transformed token. Three scores ask whether the contrast is consistent and substantial: absolute mean difference divided by root-mean-square difference; fraction of differences with the mean's sign; and total squared difference divided by the two tokens' combined squared loadings. The last ratio can exceed one; it is not a fraction of total tensor energy. Bars are 0.5, 0.8, and 0.1, respectively. The algebra/metric checks agree within $1.33\times10^{-13}$, and centering the token loadings leaves every difference unchanged.

Twenty-four branch/relation cells qualify. Parent 1 branch 0 prefers `s`, `es`, and `ies` forms on **98.45%, 98.80%, and 100%** of pairs, with mean/RMS ratios 0.886, 0.911, and 0.905. Branch 1 prefers `ing` forms on 89.03% of pairs. Different parents also qualify, so atlas A/B/C pass. These are properties of output weights; the scalar input product can reverse the write in context.

## Red-team: broad categories and small components

The mean pair difference is unchanged by permuting the transformed tokens:

$$
\frac1N\sum_i(f_{b_{\pi(i)}}-f_{a_i})=\bar f_b-\bar f_a.
$$

Thus a strong mean can describe a broad suffix category without relating each word to its specific inflection. Averaging over every independently chosen source and transformed token gives the exact squared-difference expectation

$$
\mathbb E_{i,k}(f_{b_k}-f_{a_i})^2
=\overline{f_b^2}+\overline{f_a^2}-2\bar f_a\bar f_b.
$$

The [executed red-team](BRANCH_TOKEN_RELATIONS_REDTEAM_V1.json) passes these identities within $5.21\times10^{-18}$. For parent 1 branch 0, matched differences retain 95.6–97.0% of the independent-pair energy, missing the proposed 20% reduction in all three suffix families. Most of its output regularity is therefore a broad category difference, with only a small reduction in variation from lexical matching. This does not refute a useful suffix preference.

Unit writer normalization can also highlight tiny branches. Only 19/24 atlas passes retain at least 5% of their own parent's coefficient energy: 79.17%, just below the registered 80% bar. Parent 1 branch 0 is not such a tiny component; it retains 53.67% of its node energy. Fractions of overlapping standalone node banks must not be added as fractions of the native layer. Cross-parent writer cosines reach 0.923, but output similarity alone neither identifies the same input computation nor establishes causal reuse.

## Frozen grammatical screen

The next screen left all factors fixed and tested 16 verbs and 16 nouns. Subject–verb prompts contrast `he` and `they`, after a sentence containing the corresponding base verb. Count-noun prompts contrast `one` and `two`, after a sentence naming the plural noun. Base/donor orientation alternates, so both directions are tested. A singular subject favors a suffixed verb, whereas a plural count favors a suffixed noun: the two tasks require different context-to-form mappings.

Controls comprise 16 answer-preserving pronoun substitutions (`he`/`she`, `they`/`we`) and the 16 historical V6 unrelated controls. One proposed verb, `swims`, failed the single-token check before any model outcome; the V2 row builder substitutes `wait`/`waits`. The failure and repair are preserved. No native-outcome filtering occurred. [Frozen rows](FROZEN_BRANCH_MORPHOLOGY_V1_ROWS.json) · [Row-builder failure](BRANCH_MORPHOLOGY_ROWS_V1_FAILURE.json).

The managed run completed at 20:21:43 with 20 body forwards, 144 sequences and 1.50 seconds inside the experiment. All native agreement/count/pronoun-control endpoints prefer the specified answer over its foil. Physical intervention replay and native capability pass. Branch 0's direct write supports the donor answer on all 32 task pairs. Controls pass with mean absolute CE changes **0.00619** and **0.00416** nats. These are close controls on a small constructed screen, not comprehensive collateral or OOD guarantees.

For base final state $h_b$, branch amplitudes $a_b,a_d$ and physical writers $V$, the intervention is

$$
h_{\mathrm{swap}}=h_b+V(a_d-a_b).
$$

The full native RMS normalization and capped logits are then recomputed. Transfer divides the change in donor-answer-minus-foil margin by the corresponding complete native donor-versus-base margin gap. Results are:

| Task / direction | Bank margin shift | Full native gap | Recovery |
|---|---:|---:|---:|
| Verb base → suffixed | 0.5522 | 12.2591 | 4.50% |
| Verb suffixed → base | 0.5231 | 11.6337 | 4.50% |
| Noun base → suffixed | 0.3780 | 8.9302 | 4.23% |
| Noun suffixed → base | 0.3356 | 9.3497 | 3.59% |

The 0.05 absolute-margin criterion holds, but the 10% recovery criterion fails in every task/direction. The full verdict is **A/B/C/E pass; D fails**. A useful, directionally correct small effect is not sufficient extraction. [Managed result](FROZEN_BRANCH_MORPHOLOGY_V1_RESULT.json).

## Does the native last MLP have the missing signal?

The run also saved native-MLP-output-only swaps. A subsequent exact two-token CPU replay distinguishes their local scope from the full-model donor gap, using the same method as the earlier tense control. It retains the full state norm and tanh and agrees with the GPU within $5.10\times10^{-6}$ nats. [Receipt](FROZEN_BRANCH_MORPHOLOGY_LOCAL_ANATOMY_V1.json).

Native last-MLP-only swaps transfer **25.2–33.0%** of the full-model margin change across the four task/direction means. The two branches reproduce only **12.8–17.9% of those local MLP swap effects**, missing the separate 50% local-coverage bar. Holding the base normalization denominator fixed changes these effects only modestly. Local accounting A/B pass and C fails. These alternate denominators do not change the original screen verdict.

Unlike the tense screen, there is a sizeable positive native MLP contribution to explain. The next weights-first question is whether folding the broad suffix contrast through the **entire native last bilinear layer**, rather than just this approximate node, reveals a larger compact computation. Such a fold would preserve bias, radial terms and final normalization interfaces. It would test a new output readout of the original weights, not fit the current validation examples or redefine the present small bank as sufficient.

## Full-native readout fold: absolute values and changes differ

That next fold is now executed for all six spelling relations. For each relation, average the transformed-minus-base unembedding rows and normalize the resulting physical readout $r$. Its native scalar MLP output is

$$
f_r(x)=x^\top Q_rx+\beta_r,\qquad
Q_r=\operatorname{sym}\!\left(L^\top\operatorname{diag}(D^\top r)R\right),
\qquad \beta_r=r^\top b.
$$

Here $R$ in the equation denotes the native right input matrix. The trace term $\alpha_r\lVert x\rVert^2$, with $\alpha_r=\operatorname{tr}(Q_r)/1152$, and the bias are retained exactly. Complete eigenanalysis of the traceless remainder gives the optimal signed-square approximation. One real product can retain the largest positive and largest-magnitude negative eigenvalues together; $M$ products can retain up to $M$ eigenvalues of each sign. These are scalar coefficient-Frobenius optima at fixed readouts, not general circuit or arbitrary-output-mixture lower bounds.

| Fixed suffix readout | One-product capture | 16-product capture | Products for 90% |
|---|---:|---:|---:|
| `s` | 19.81% | 45.44% | 202 |
| `es` | 16.30% | 39.61% | 215 |
| `y` → `ies` | 16.10% | 38.48% | 216 |

All exact folding and leading-product checks pass within $4.11\times10^{-15}$. The one/16-product 90% bars fail. This is not an iterative optimization failure. It also does not establish that the large coefficient remainder matters equally on native model states. [Fold, spectra and compact factors](NATIVE_TOKEN_RELATION_FOLD_V1.json).

We therefore evaluated the already frozen leading product and **16 signed squares** per readout on 384 FineWeb, 384 corpus-shift and 128 grammatical-screen endpoints, with no data fitting or rank selection. Sixteen squares and sixteen general products are different budgets; the natural-input test uses the saved sixteen-square approximation. Exact folding replays cached native MLP outputs within $1.03\times10^{-6}$ relative error. The first validation script stopped on an old receipt's nested cache-hash schema; V2 repairs that lookup only, with the same factors, computations and bars. [Preserved failure](NATIVE_TOKEN_RELATION_VALIDATION_V1_FAILURE.json) · [Completed validation](NATIVE_TOKEN_RELATION_VALIDATION_V2.json).

The general-text absolute-value bar fails. Suffix full-output relative RMS errors are **35–52% on FineWeb** and **39–57% on the corpus-shift panel**. The retained trace/bias terms do not make these accurate general predictors.

However, **paired grammatical changes pass**: all three readouts predict the native scalar differences with **0.93–8.43% relative RMS error**, and every pair has the correct sign. For the `s` readout specifically, the errors are 0.93%/1.38% for the two subject–verb directions and 6.43%/5.45% for count nouns. Prediction of a difference can be much better than prediction of its endpoints when the approximation error is shared between paired contexts. These results validate frozen weights on the existing constructed panel; they are not fresh OOD confirmation or a physical circuit replacement.

The next physical screen uses the three suffix readouts as rows of a matrix $A$. Their dual output writers are

$$
V=A^\top(AA^\top)^{-1},\qquad AV=I.
$$

Consequently, $VA$ is the orthogonal projector onto their joint physical readout span. Using dual writers avoids simply adding three overlapping projections. The compact scalar program supplies 16 signed squares per readout, 48 total, plus the exact radial/bias terms. The native complement remains required. Its donor-difference write is tested separately from absolute replacement, so accurate differences cannot conceal an inaccurate ordinary replacement. The source is bound and queued through the managed runner; no physical-screen result is claimed in this paragraph.

### Physical screen completed: useful changes, failed ordinary replacement

The managed tail-only screen completed at 20:34:48: zero model-body forwards, 512 tail rows and 0.485 seconds inside execution. The readout Gram condition number is 24.76; the dual-writer identity agrees within $5.24\times10^{-16}$. Earlier native margins and the old bank intervention replay within $5.73\times10^{-6}$ nats. [Physical result](NATIVE_RELATION_PHYSICAL_V1_RESULT.json).

| Task / direction | Exact native span recovery | 48-square recovery | Relative error in intervention effect |
|---|---:|---:|---:|
| Verb base → suffixed | 19.13% | 19.84% | 3.93% |
| Verb suffixed → base | 19.77% | 20.68% | 4.91% |
| Noun base → suffixed | 21.96% | 20.58% | 6.31% |
| Noun suffixed → base | 18.79% | 17.77% | 5.41% |

Recovery uses the original full-native donor margin gap. The approximation's intervention-effect signs agree with the exact span on every task pair. Mean absolute swap CE on pronoun and unrelated controls is 0.00804 and 0.00590 nats, respectively. **A/B/C/E pass:** native-span transfer, approximate-effect fidelity and these controls hold. This is a stronger change predictor than the previous two-branch surrogate, with no factor fitting to the prompts.

**D fails:** replacing the native three-coordinate output with its approximate absolute output changes count-noun CE by 0.15930 nats on average, above the 0.05 allowance. Verb/pronoun replacement changes are 0.03908 and unrelated-control change is 0.01304. We cannot adopt this as an ordinary layer replacement or claim that it supplies the whole grammatical computation. Native complementary computation remains installed. The tested object is a compact, conditional predictor of a substantial signed causal change.

A follow-up tests the tempting explanation that all omitted computation is merely constant across each pair. Let $e_b=f(x_b)-\widehat f(x_b)$ and $e_d=f(x_d)-\widehat f(x_d)$. Split them into

$$
e_{\mathrm{common}}=\tfrac12(e_b+e_d),\qquad
e_{\mathrm{difference}}=\tfrac12(e_d-e_b).
$$

Using the dual-writer metric $V^\top V$, the summed endpoint error energy is exactly twice the sum of these two energies. The [executed accounting](NATIVE_RELATION_PAIR_REMAINDER_V1.json) holds within $2.20\times10^{-16}$. However, the difference component carries 11.1–11.9% of paired error energy on verbs and 3.2–4.1% on nouns, above the registered 1% bar. Base/donor error cosine is only 0.76–0.78 on verbs, though 0.97 on nouns. A passes, B/C fail. Shared error contributes to the difference/absolute split, but the stronger invariant-offset explanation is unsupported. No empirical offset was fitted or inserted.

The next priority is a frozen test on different lexical items and grammatical constructions, followed by explicit removal/composition checks if transfer survives. General-text prediction and ordinary replacement remain separate unresolved requirements. Successful task differences do not exhaust the native tensor or establish a unique hierarchy.

One final weight-only audit identifies a narrower approximation detail to check before interpreting replacement failure: truncating a traceless matrix does not preserve its trace. The original explicit radial term was retained, but the retained sixteen-square remainder adds trace -33.17, -199.00 and -3.46 for the three suffix readouts. The known trace-preserving repair subtracts this retained trace times $\lVert x\rVert^2/1152$ from the approximation. Its algebraic trace identity passes exactly; it has not yet been evaluated as a corrected physical replacement. This correction comes from weights, not an empirical offset, and is existing radial-repair methodology applied to this new program. It leaves donor differences unchanged when input radii are equal. [Trace-leak audit](NATIVE_RELATION_TRACE_LEAK_V1.json). Preserve every existing replacement/prediction failure until that separate check is executed.

### Trace correction completed without repairing replacement

The separate trace-preserving physical replay now passes A/B/C/E and still fails D. Count-noun replacement mean absolute CE is **0.16510**, slightly worse than the previous 0.15930. Verb/pronoun replacement changes are 0.03810 and unrelated-control change is 0.01833. Swap effects and their successful approximation/control verdicts are essentially unchanged. The trace correction replaces three existing radial coefficients; it adds no factors and fits no data. This closes the trace-leak explanation for the observed replacement failure. [Physical replay](NATIVE_RELATION_TRACE_PHYSICAL_V1_RESULT.json).

The matching general-text scalar validation still fails its absolute prediction bar, while paired-change prediction continues to pass. That result preserves the distinction between a useful change predictor and an accurate ordinary replacement. [Scalar validation](NATIVE_TOKEN_RELATION_TRACE_VALIDATION_V1.json).

### Frozen construction holdout and a smaller-component check

A new panel is frozen with 16 different verbs and 16 different nouns; all task answer/foil token IDs are disjoint from the earlier panel. Subject–verb prompts use a noun head followed by either an intervening plural noun or a relative clause, with construction and swap direction crossed. Count prompts use `a single`/`several` in place of `one`/`two`. Adjective changes provide answer-preserving controls; the historical unrelated controls are retained and explicitly labelled. The same trace-corrected 48-square program is queued through the managed runner, with no factor changes or outcome filtering. These are new validation constructions and lexemes, not proof of corpus or pretraining disjointness. [Frozen rows](NATIVE_RELATION_HOLDOUT_V1_ROWS.json).

While that job waits behind the peer's live run, an independent CPU test evaluates the already saved **one real product per readout**, three products total. It retains the correct trace and bias. On the original panel, the small program preserves the sign of every scalar change, but its joint physical-write relative error is **18–19% for verbs and 42–44% for count nouns**. Scalar errors likewise miss the 25% bar for the noun `s`/`es` readouts. A passes; B/C fail. Thus a three-product simplification is not an adequate substitute for the demonstrated two-task 48-square change predictor under the registered criterion. It was not fitted or rescaled on these data. [Original-panel result](NATIVE_RELATION_ONE_PRODUCT_V1_ORIGINAL.json).

### Construction holdout completed successfully for changes

The managed holdout completed at **20:49:59**, after the peer's live job finished. It used 20 body forwards, 144 sequences and 1.48 seconds inside execution. All task and adjective-control endpoints pass native answer-versus-foil capability. Instrument, capability, effect approximation, transfer and swap-control bars pass: **A/B/C/D/E true; ordinary replacement F false**. No rows were filtered by outcomes. [Completed result](NATIVE_RELATION_HOLDOUT_V1_RESULT.json).

| New task / direction | Exact native span recovery | Frozen 48-square recovery | Relative effect error |
|---|---:|---:|---:|
| Verb base → suffixed | 16.59% | 17.16% | 3.76% |
| Verb suffixed → base | 16.38% | 17.10% | 4.25% |
| Noun base → suffixed | 18.10% | 16.60% | 8.67% |
| Noun suffixed → base | 13.69% | 12.40% | 9.47% |

All task intervention-effect signs match the exact native span. Mean absolute approximate-swap CE changes are **0.02920** on adjective controls and **0.00590** on the reused unrelated controls. Relative-error/sign fidelity is registered for the task effects, not for the small control effects; some control-relative errors are large despite their small absolute CE changes.

Descriptive construction-specific accounting also retains the result: plural-attractor prompts recover 19.01% of the native gap with 3.55% effect error; relative-clause prompts recover 14.14% with 5.75% error; single/several noun prompts recover 14.44% with 8.99% error. These are eight, eight and sixteen pairs respectively, without a separately registered confidence claim. The fixed program has therefore predicted useful signed effects across new lexical items and constructions. This is evidence for that limited form of generalization, not proof of corpus OOD, pretraining disjointness, complete extraction or the entire four-property goal.

Ordinary replacement still changes CE too much: mean absolute changes are **0.08663/0.09459/0.08292/0.01833** for verb, noun, adjective and unrelated families. All original replacement failures remain. The trace-corrected program is useful for the tested changes, not an adopted replacement.

The prospectively specified three-product CPU check was then executed on the new cache. Algebra passes; the simplification criteria still fail. Physical-write relative error is about **15% for verbs and 53–55% for nouns**, with all scalar signs correct. [Holdout one-product result](NATIVE_RELATION_ONE_PRODUCT_V1_HOLDOUT.json). These are the saved coefficient-optimal leading products, not the best possible three-product task predictor. The differing errors motivate a concrete next reuse test: separate those leading products from the remaining fixed quadratic terms and measure whether a shared component plus additional terms predicts the two behaviors' effects and their composition. It does not yet establish that split as a semantic hierarchy.

## Exact split, composition and removal

The next test is completed. Each of the three scalar functions is split into its largest positive/negative eigenpair and the remaining fourteen signed squares. The eigenpair becomes one real product using the existing signed-square pairing identity. The original radial term and bias belong to the leading part; both parts receive their specified trace corrections. Their sum is exactly the prior trace-corrected program. This assignment matters for removal and is frozen before its measurement.

The resulting executable artifact uses **three general products plus 42 squares**, or 45 variable products, with the same 48 dense input readers. The saving of three products is a known algebraic rewrite, not a new circuit discovery. Independent leading-eigenpair, trace and full-execution checks agree within $2.05\times10^{-15}$, including both native input panels. [Compiler and artifact receipt](NATIVE_RELATION_SPLIT_V1.json).

Write the physical outputs of these parts as $C(x)$ and $R(x)$. Their individual donor swaps add $\Delta C=C(x_d)-C(x_b)$ or $\Delta R=R(x_d)-R(x_b)$ to the same native base state. The joint swap adds both. Removal subtracts the relevant part at its own endpoint; it does not zero the entire native MLP. The native complement remains installed and required.

The managed screen used zero body forwards, 1,408 tail rows and 0.55 seconds inside execution. Earlier full-program margins replay within $6.20\times10^{-6}$ nats. **A/B/E pass; C/D fail.** [Full result](NATIVE_RELATION_COMPOSITION_V1_RESULT.json).

The leading part supports both grammatical changes. The remaining terms contribute different fractions of the whole swap effect:

| Panel | Remainder fraction, verbs | Remainder fraction, nouns |
|---|---:|---:|
| Original | 21.32% | 30.41% |
| Construction holdout | 7.48% | 26.98% |

The preregistered requirement was at least 20% for nouns and a noun-minus-verb difference of at least ten percentage points on **both** panels. The original difference is only 9.08 points, so C fails. These measurements suggest differing contributions, but do not establish the proposed semantic split or a uniquely identified shared hierarchy.

### Removal has a measurable and limited selective effect

Removing the whole component at both endpoints reduces their grammatical margin contrast by **18.7–20.8% on the original panel** and **13.4–17.7% on the holdout**, exceeding the registered 10% criterion in every task/direction cell. This is attenuation of the counterfactual contrast, not erasure of the behavior or proof that every token's CE worsens.

Mean absolute CE changes on the unrelated controls are **0.03884** for leading-part removal, **0.03172** for remainder removal, and **0.00506** for whole-component removal. All pass 0.05. The same sixteen unrelated controls are reused on both panels, so this is not an independent replication of collateral preservation. Their partial cancellation also shows why whole-component preservation does not imply that every part is harmless alone.

Adjective/pronoun controls preserve the *same grammatical computation*. They are appropriate controls for answer-preserving swaps, but not unrelated behaviors that removal should necessarily preserve. Their removal effects are reported rather than silently excluded from the receipt. Broader controls involving other inflections and nearby grammatical behaviors remain needed before a strong selectivity claim.

### State composition and CE addition are different claims

The two physical writes sum exactly by construction. Their answer-margin effects are nearly additive: mean absolute nonadditivity is **0.11–0.45%** of the whole effect across tasks/panels. However, original-panel cross-entropy nonadditivity is **0.09462/0.12765** nats for verbs/nouns, above the registered 0.05 threshold. Holdout values are 0.01930/0.03399. Therefore D fails, even though the actual composed state and its native tail were evaluated directly.

A subsequent CPU red-team uses the known cross-entropy identity. For capped logits $z_0,z_C,z_R,z_{CR}$, define the hypothetical additive-logit vector

$$
z_{\mathrm{add}}=z_C+z_R-z_0.
$$

Then the measured CE cross-term splits exactly into

$$
\begin{aligned}
&\mathrm{CE}(z_{CR})-\mathrm{CE}(z_C)-\mathrm{CE}(z_R)+\mathrm{CE}(z_0)\\
={}&\underbrace{\mathrm{CE}(z_{\mathrm{add}})-\mathrm{CE}(z_C)-\mathrm{CE}(z_R)+\mathrm{CE}(z_0)}_{\text{loss curvature under additive logits}}\\
&+\underbrace{\mathrm{CE}(z_{CR})-\mathrm{CE}(z_{\mathrm{add}})}_{\text{additional nonlinearity in the model tail}}.
\end{aligned}
$$

The split identity holds within $2.23\times10^{-16}$ and GPU CE cross-terms replay within $1.09\times10^{-5}$ nats. The tail correction has mean absolute magnitude **0.00127–0.00602 nats** across task/panel cells and only **3.9–7.1%** of the measured cross-term's RMS. The separately registered accounting A/B/C bars pass. Most of the failure of scalar CE addition is therefore accounted for by the loss function's curvature. No loss correction or empirical offset was fitted, and the original D failure remains. [Accounting and receipt](NATIVE_RELATION_CE_CURVATURE_V1.json).

The supported object is now a fixed, executable weight-derived component with tested cross-construction change prediction, a limited selective-removal result, and explicit joint execution of its parts. It is **not** an adopted replacement, the full text-to-answer circuit, a broadly OOD-validated model, or a completed four-property decomposition. The next priority is stronger neighboring-behavior collateral tests rather than further rank reduction or relabeling the failed specificity criterion.

## Neighboring inflections expose the limit of selective removal

The broader control screen completed at **21:12:40 UTC**. The same frozen component was removed on 32 pairs spanning 16 verbs, with plural subjects throughout. One family contrasts `Every day they walk` with `Yesterday they walked`; the other contrasts `They often walk` with `They are currently walking`. Each prompt first introduces its verb in an infinitive. These choices test nearby inflections without requiring singular-subject `-s` agreement. The two families share their 16 lexemes; they are not 32 independent lexical tests. No examples or factors were selected by model outcomes. [Frozen rows](NATIVE_RELATION_NEIGHBOR_V1_ROWS.json).

The model prefers the specified answer over its foil on all past-family endpoints and all base-form progressive endpoints; it succeeds on 15 of 16 progressive-inflected endpoints. Every family/side exceeds the registered 85% capability bar. Hooked native and removal execution agrees with manual tail execution within $7.47\times10^{-7}$ relative logit error. Instrument and capability pass, so this is a valid control test. [Physical result](NATIVE_RELATION_NEIGHBOR_V1_RESULT.json).

| Removal measurement | Past versus base | Progressive versus base |
|---|---:|---:|
| Whole-component mean absolute target CE change | 0.07078 nats | 0.05948 nats |
| Leading-part removal, same metric | 0.07832 | 0.06368 |
| Remainder removal, same metric | 0.02155 | 0.02402 |
| Whole-component attenuation of grammatical contrast | 4.17% | 3.21% |
| Exact native readout-span removal, mean absolute CE change | 0.06105 | 0.05258 |

The preservation bar was 0.05 nats and at most 5% absolute contrast attenuation, separately for each family. The contrast criterion holds, but whole-component CE preservation fails on both families. The separate requirement that both partial removals preserve CE also fails because the leading part exceeds the allowance. **A/B pass; C/D fail.** Averaged signed CE changes are much smaller because improvements and damage cancel; they cannot replace the absolute-preservation criterion.

Compared with the previously measured 13–21% attenuation of the intended agreement/count contrasts, 3–4% on these neighboring contrasts suggests relative selectivity on the tested panels. It does not establish clean isolation, a universal selectivity ratio, or preservation of all other behavior. Removal of the *exact native* three-readout component also exceeds 0.05 nats on both families: approximation error alone cannot explain the spillover. That exact-span arm is a descriptive control, not a separately registered success.

### Red-team: is this mainly final normalization?

The CPU successor recomputes both full-vocabulary tails from the cached native state $h$ and the removed write $w$. The actual and fixed-denominator logits are

$$
z_{\mathrm{actual}}=30\tanh\!\left(\frac{U(h-w)}{30\rho(h-w)}\right),
\qquad
z_{\mathrm{fixed}}=30\tanh\!\left(\frac{U(h-w)}{30\rho(h)}\right),
$$

$$
\rho(h)=\sqrt{\frac{1}{1152}\lVert h\rVert^2+\epsilon},
\qquad \epsilon=1.1920928955078125\times10^{-7}.
$$

Holding the denominator fixed isolates the write's effect through the unembedding and logit cap. The difference between these two outcomes measures the additional effect of changing the normalization denominator. This is diagnostic accounting, not a substitute model or a repair to the failed preservation criterion.

Actual CE and margin replay agrees with the GPU within $9.54\times10^{-6}$ nats. The independent identity $\Delta\mathrm{CE}=\Delta\log\sum_v e^{z_v}-\Delta z_{\mathrm{answer}}$ holds within $6.22\times10^{-15}$. For whole-component removal, the denominator correction averages **0.00388 / 0.00364 nats** in absolute magnitude for past/progressive, respectively; it is only **4.52% / 6.29%** of actual CE-effect RMS. Exact-span removal has similarly small corrections. The separately registered accounting A/B/C all pass. [CPU result](NATIVE_RELATION_NEIGHBOR_NORM_V1.json).

Thus direct output writing dominates this spillover. The failed preservation claim remains failed. The useful next structural question is whether the weight-derived output functions contain a shared inflection computation plus separable branches, rather than assuming that the original `s/es/ies` readout grouping already identifies an isolated number circuit. Any new grouping must be frozen before fresh behavioral validation; this control panel is evidence about the current program, not training data for a replacement.
