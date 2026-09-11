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
