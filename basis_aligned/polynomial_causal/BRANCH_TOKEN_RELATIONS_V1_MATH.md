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
