# Producer-metric spectral results

12 September2026. The [frozen derivation and predictions](PRODUCER_METRIC_SPECTRAL_V1_MATH.md) use the exact weight-derived MLP16 quadratic-function Gram to choose the outer32terms per output. This preserves the paired coefficient-matrix objective and the native producer interface. The function has two frozen output directions, not the whole unembedding. The derivation is source-bound and remains unchanged; execution updates belong here.

## Native comparison completed

[The native receipt](PRODUCER_METRIC_OUTER32_V1_RESULT.json) completes in1.66seconds, peak allocation0.94GB. All1152producer-metric directions survive; condition number79.22. Orthogonality, baseline replay and paired-error identities agree within1.8e-15. This is an exact spectral solution for its stated paired rank32 objective, not for the fully symmetrized quartic or native loss.

The ordinary outer32 paired errors0.6114/0.6169 become0.5925/0.5966. Relative error reductions3.09/3.30% **miss the registered10% improvement bar**. The behavioral bars pass: original developmental swaps0.78/5.37/3.52/1.27%, all signs correct; removal disagreements0.00411/0.00963/0.00182/0.00254nats. Overall write error1.574% versus1.694% for the ordinary baseline. The cost is identical:76,096 fitted values plus15,925,248 shared native producer values, scale, normalization and background.

## Inspected-panel diagnostic completed

[The previously inspected panel comparison](PRODUCER_METRIC_INSPECTED_PANEL_V1.json) passes all its diagnostic bars: swaps5.99/3.17/4.55/4.52%, all64signs correct; removals0.00405/0.00570/0.00357/0.00429nats. Write error improves2.362→1.582%. In particular, progressive improves10.54→4.52%. This does not retrospectively repair the ordinary program's failed prediction. The new method was proposed after that panel was inspected, so these results are not clean confirmation of its generalization. The initial report exactly reproduces the earlier panel receipt, and the fixed reference effects replay exactly.

The result supports a useful composition-aware choice of inner readers: weight the outer directions by what the native producer can write, rather than fitting arbitrary full quadratic matrices or assuming equal producer-output geometry. No language-data optimization was used. The modest coefficient gain and larger behavioral change also caution against treating this paired norm as a calibrated behavioral-error certificate.

## Frozen contextual confirmation is submitted

[The registered context test](PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_PREREGISTRATION.md) freezes both ordinary and weighted programs on512newpairs,128per task family, with8surrounding contexts and changed core constructions. It scores every one of32family/context cells separately. The16lexical groups are reused, so the512pairs are correlated; this is context/construction transfer, not new-lexeme or corpus OOD.

The managed job requests130body forwards/1040sequences,lengths14–24, including two physical/manual controls. Primary all-cell bars remain10%swap error,90%sign agreement with4livepairs,0.02-nat removal disagreement,5%write error and75%native capability on both sides. The ordinary program is a matched comparator. A baseline failure cannot rescue a weighted-program failure, and an aggregate average cannot replace the all-cell criteria. Check the authoritative result/runner for execution status; queue acceptance itself supplies no evidence of fidelity.

## Shared producer functions: overlap does not justify merging

While the contextual test waits behind a verified live shared-runner job, the [CPU function-space probe](PAIRED_PRODUCER_SHARED_SPACE_V1.json) measures overlap between the two frozen32-reader banks using the exact producer Gram $H$. No individual reader pair has absolute correlation>=0.9. Allowing mixtures within each bank gives principal cosines0.985,0.961,0.925 for the leading three directions; none reaches0.99. Mean squared principal cosine is0.371. The uniform random32D-subspace expectation in1152dimensions is0.0278, provided as a geometric reference, not a significance test or a model of these learned matrices.

For each aligned unit-variance pair $g_0,g_1$ with correlation $c$, define common and difference functions

$$
h=\frac{g_0+g_1}{\sqrt{2+2c}},\qquad
 d=\frac{g_0-g_1}{\sqrt{2-2c}}.
$$

Then both branches can read the same two functions:

$$
g_0=\sqrt{\frac{1+c}{2}}h+\sqrt{\frac{1-c}{2}}d,\qquad
 g_1=\sqrt{\frac{1+c}{2}}h-\sqrt{\frac{1-c}{2}}d.
$$

The [actual native graph check](PAIRED_COMMON_DIFFERENCE_GRAPH_V1_RESULT.json) applies this to the fixed leading three pairs and retains the other29directions per bank. Exact reconstruction agrees within2.3e-15. It still uses64parent functions and adds branch adapters:78,150 versus76,096 fitted values. This is an exact shared interface, not a simpler or semantically identified decomposition.

Dropping the three difference functions is a separately registered approximation. It misses every substantive preservation bar: paired branch errors12.31/12.64% versus5%, write change3.12% versus1%, and native removal disagreements0.020009/0.027362nats for agreement/count versus0.02. Swaps still pass. The potential61-parent payload would have74,694 values, but no standalone merged program is adopted; the saved analysis artifact contains the exact graph and validation writes. Even that potential reduction is only1,402values before the unchanged15.93million-value native producer, under0.01% of the combined payload.

The executed [cross-term accounting](PAIRED_COMMON_DIFFERENCE_CROSS_TERMS_V1.json) identifies why the merge loses information: **99.44/99.51% of its paired coefficient error comes from mixed retained/difference products**, not from squared difference functions. For one squared root, the coefficient of $hd$ is $\sqrt{1-c^2}$, equal to0.173/0.275/0.379 for these three correlations. Thus a small difference in an input function can remain important after multiplication with a shared function. The full branches also mix these coordinates with other parents; the matrix-level accounting includes all those interactions and reproduces the measured error exactly.

This is useful evidence for interaction-based graphs: keep common computations and the products involving their differences, rather than merging approximately aligned readers solely by cosine. It does not prove that joint graph optimization cannot find a better basis, or that these variables are semantic circuits. The fixed merge's failure is preserved; no merge-size sweep or label-driven repair has been run. The contextual confirmation of the unmodified weighted program remains a separate queued experiment.

## What the partial component writes—and what fidelity has not established

The [output-space description](PAIRED_OUTPUT_SPACE_DESCRIPTION_V1.json) examines centered token loadings $UW$ of the fixed two-dimensional write basis. Top16/64/128/1024vocabulary rows contain0.495/1.576/2.756/14.191% of its loading energy. This is diffuse, not a tight token cluster. Some leading individual rows are formatting symbols or byte fragments; unused output rows contain0.791% of the energy. These observations do not justify a frequency-based or linguistic name. Individual basis coordinates can rotate; the reported row-energy ranking is rotation invariant. The selected morphology contrast directions have descriptive coherence0.62–0.84, not causal identification.

The [loss-partition analysis](PAIRED_OUTPUT_CE_PARTITION_V1.json) evaluates removal of the **exact reference component** on the already inspected lexical panel. For answer/foil logits $z_a,z_f$,

$$
\mathrm{CE}(a)=\operatorname{softplus}(z_f-z_a)
+\log\sum_v e^{z_v}-\log(e^{z_a}+e^{z_f}).
$$

The first term measures the binary contrast; the second concerns probability outside that pair. Subtract the native baseline from each term to obtain an exact removal-effect partition. Native logits are evaluated in float32, followed by float64 loss accounting; the identity holds within5.4e-15. This precision choice differs slightly from the original float32-loss effect receipts and does not replace their scores.

| Family | Mean signed full CE damage | Binary-contrast part | Outside-pair part |
|---|---:|---:|---:|
| Agreement |0.19713|0.01838|0.17875|
| Count |0.19684|0.02998|0.16686|
| Past |0.11251|0.00522|0.10729|
| Progressive |0.16969|0.01122|0.15847|

The outside-pair part accounts for84.8–95.4% of the **signed mean** damage, not that fraction on every example or a variance decomposition. Binary damage is positive on only34.4–46.9% of endpoints; the means include heterogeneous signs. Thus accurately reproducing this component's removal mostly reproduces a broader vocabulary-distribution contribution on this panel. It does not, by itself, establish a selective morphology circuit.

The full goal remains substantially open:

- Prediction: developmental effects pass; the first new lexical panel failed for the ordinary program. The weighted program's inspected-panel pass is diagnostic; its new contextual confirmation is pending at this entry.
- Extraction: the executable scalar-quartic interface is specified, but exact native MLP16 and normalization/background dependencies remain. The hook screen evaluates native background plus approximation minus exact reference. It does not remove the original native machinery from a compiled full model.
- Removal: effect replication is tested; semantic selectivity and preservation across unrelated behaviors are not established by these four families alone.
- Composition/reuse: exact shared-coordinate algebra is checked, while the attempted simpler merge fails. Joint adoption with another independent extracted program and stable identification remain untested.
- Structural simplicity:76,096 fitted values describe this approximation conditional on15.93million shared producer values. The entire teacher background also remains in the test harness. There is no claimed whole-model parameter reduction or end-to-end cheaper deployment.

The handoff's updated criterion requires a previously unspecified reusable computation with explicit consumers and fewer independently specified computations. These results provide a controlled approximation and constraints on graph discovery; they do not satisfy that criterion merely by reproducing a selected two-output projection.

## Contextual confirmation completed: five swap cells fail

The [512-pair result](PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_RESULT.json) completes130body forwards/1040sequences in7.42seconds excluding binding checks. Numerical A passes: physical/manual controls<=7.0e-7 and composed algebra<=3.9e-15. All32weighted removal and write-error cells pass; maximum disagreement0.010945nats and maximum write error1.957%. Registered B and E fail.

Weighted swaps pass27/32cells, compared with20/32for the ordinary baseline. The weighted failures are count/context1 at10.95% and progressive/contexts0,1,3,6 at15.78/10.58/12.66/14.41%. All failed cells remain in the result. This improves contextual fidelity but does not pass the registered all-cell prediction.

Native capability fails in six past-tense base cells: contexts0,2,4,5,6,7 have0–12.5%correct proposed base-answer contrasts. The preceding past-tense wording can support a narrative reading, so this is a failure to establish the intended native capability, not a claim that the model violates an unambiguous grammar rule. No examples were relabelled or deleted. The [cell audit](PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_CELL_AUDIT.json) shows these failures are **disjoint** from the five weighted swap failures; they cannot explain away B. An all-cell criterion is not replaced by an average over favorable contexts.

The next experiment broadens the weight-first object to the [full-unembedding paired producer path](FULLU_PAIRED_PRODUCER_V1_PREREGISTRATION.md), reusing the established full-U output-function spectrum with the producer metric. Its small explicit-tensor control passes; native execution is submitted separately. The point is to test structure and sufficiency for the complete interaction path, rather than continuing to tune this selected two-output slice. No circuit is promoted from these contextual results.
