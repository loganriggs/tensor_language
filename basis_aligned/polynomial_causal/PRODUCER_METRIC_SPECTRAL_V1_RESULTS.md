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
