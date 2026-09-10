# Joint unembedding and QK experiments — through 19:14 UTC

**Both requested directions ran.** The unembedding results reject two particular simple weight structures. The corrected QK experiment finds partial within-frame transfer but fails the full separation claim. Neither result identifies a four-property circuit. The main unembedding search is now explicitly **unsupervised joint factor discovery**, as the user clarified; the four-word example below was a test of the math file's illustrative program.

## Entire-unembedding primitive structure, 19:00

All50,304 model output rows and all4,608 native input products were included. Exact full-vocabulary output-code Gram and input-product Gram avoided a huge interaction tensor. Negative scales were allowed. There were zero qualifying near-duplicate input products with different consumers, and zero qualifying near-identical signed output profiles with different input products, under fixed10%substitution bars. Closest input-product error was65.3%; closest centered output-profile error71.9%. This does not test arbitrary shared linear factors or recombinations of products. Instrument held to3.5e-15;0native forwards,.9924executor seconds.

## Source-file joint family example, 19:12

Following [the user-provided math](unembedding_folding_in_math.md), we formed the four exact token quadratics for “ is”, “ are”, “ was”, “ were”. We tested present/past means $P,T$, number contrasts $N_p,N_t$, and their average $N$. A three-product program predicts $P^*+N^*,P^*-N^*,T^*+N^*,T^*-N^*$, where each star is the best single-real-product approximation of that combination.

The difference between $N_p$ and $N_t$ was70.1%relative to their RMS Frobenius norm. Joint errors were84.95%,86.32%,88.57%,88.95%. Leading mean factors have absolute cosine up to.9767, but their single-product reconstructions each miss85–88%, so this is not a sufficient shared-factor program. Instrument held near3e-15;0native forwards,1.1462executor seconds.

The saved-state CPU audit asks a broader question. A sum of $k$ real linear-form products has matrix rank at most $2k$. The eigenvalue tails of the four exact token matrices imply **at least379,383,387,391 products**, respectively, to reach10%full-input coefficient error. These are numerical spectral necessary bounds in the proposed sum-of-products class. They are not behavioral lower bounds, not statements about a restricted natural-input domain, and not proof that many factors cannot share linear readers or output consumers. Even allowing arbitrary rank-six approximations leaves77.8–82.5%error, so the failure is broader than our particular three fitted products.

Full-vocabulary writes for the proposed component were retained as explicit residual writers and all50,304 output coefficient rows. No native removal was run after the cheap sufficiency hypothesis failed.

## Corrected joint QK test, 19:07

Both QK factors participate through

$$
\phi_q=\operatorname{vec}(q_1q_2^\top),\quad
\phi_k=\operatorname{vec}(k_1k_2^\top),\quad
s=\phi_q^\top\phi_k/128^2.
$$

We fitted separate query/key joint-feature change spans on the first8examples of each behavior, using all26existing heads. Last8examples were excluded from fitting; they were already opened task examples, not new OOD data. Interventions projected donor-minus-current changes into those spaces while retaining both factors' joint product and its mixed terms. Donor values were held identical across comparisons. The reference was the *incremental routing correction above value-only interchange*.

| Evaluation | Own-family routing error | Other-family routing-effect magnitude |
|---|---:|---:|
| Correlative, original frame |15.33%|36.90%|
| Correlative, longer frame |62.82%|10.15%|
| Disjoint behavior |27.17%|61.38%|

Own error needed<=20%, and other-family effect<=20%, in every panel. Both joint claims fail; native capability and instrument pass. This differs from the old whole-half test: it actually tests joint-product subspaces. However, projected product features need not be realizable by a single native q1/q2 pair, so the intervention is a compiled feature edit rather than a native input-vector/weight edit.

A further representation issue remains: these bases were fitted **after positional rotation**. The correlative query moves from position5 to8 across frames. That can change the coordinates even for a shared input computation. The next controlled comparison must remove/transport those position maps before treating cross-frame failure as semantic. All live normalization factors stay explicit.28native forwards224seq,6.812executor seconds; paired intervals are in the shared CPU audit.

## Next unsupervised decomposition

The target is the full family $T_{vij}$, approximated by shared products

$$
\widehat T_{vij}=\sum_r A_{vr}\,\operatorname{sym}(a_rb_r^\top)_{ij}.
$$

Input factors are learned jointly; no token family determines them. Output coefficients reveal consumers afterward. We implemented and checked the exact implicit fitting loss and the least-squares writer update without building $T$. The native joint optimizer has **not** run yet. A bounded discovery block must retain its unexplained remainder and be compared with native products; coefficient fit alone is not identification. See [current discovery plan](../../UNSUPERVISED_JOINT_QUADRATIC_DISCOVERY_PLAN.md).

Receipts: [full-U primitive screen](../../FULL_UNEMBEDDING_SUBTERMS_V1_RESULT.json), [joint family](../../UNEMBEDDING_JOINT_FAMILY_V1_RESULT.json), [joint QK](../../CORRELATIVE_JOINT_QK_SUBSPACES_V1_RESULT.json), [CPU bounds and intervals](../../JOINT_COMPOSITION_RESULTS_V1_AUDIT.json), [implicit-fit algebra check](../../JOINT_QUADRATIC_FIT_V1_ALGEBRA_RESULT.json).
