# Why exact entity symmetry cannot meet the current query-fidelity target

On the registered original/renamed input pairs, every exactly entity-equivariant
replacement has mean query KL at least **0.01439 IID** and **0.006391 OOD**. Both
exceed the0.001 target. A smaller model is still possible, but it must retain some
identity-dependent computation rather than enforce perfect label symmetry.

The failed internal-gate transfer is a separate result. It has mean all-position
KL8.835/10.389 and query KL1.084/.848, with exact native/factorial replay around
2e-13. Those errors diagnose that particular interchange. The lower bound below
applies to every exactly equivariant predictor, independent of its implementation.

## Exact paired bound

For an input x and entity permutation sigma, let p be the native query distribution
on x. Align the native distribution on sigma(x) back to the original vocabulary:
q(j)=p_native(sigma(x))[sigma(j)]. Special tokens remain fixed. An exactly
equivariant surrogate must use the same aligned distribution r for both inputs.
Let m=(p+q)/2. Expanding the logarithms gives

    [KL(p || r) + KL(q || r)] / 2
      = [KL(p || m) + KL(q || m)] / 2 + KL(m || r)
      >= JS(p,q).

The first term is Jensen–Shannon divergence; see [Lin, 1991](https://www.cise.ufl.edu/~anand/sp06/jensen-shannon.pdf).
We use natural logarithms and report nats. The application to equivariant
surrogates follows from the explicit output-alignment constraint above. No learned
subspace, fitting, or native-error filtering is involved.

Each pair separately attains the bound at r=m. One globally consistent equivariant
model need not attain all these pairwise minima simultaneously. Therefore a low
bound means only “not ruled out by this bound,” not demonstrated feasibility.
This result does not apply to models with identity-dependent corrections.

## What the saved native outputs show

| Population | All-query paired KL floor | Hop3 paired KL floor | Native answer disagreements |
|---|---:|---:|---:|
| IID24-cycle |0.014390|0.050015|17/768 pairs|
| OODthree8-cycle |0.006391|0.025294|3/768 pairs|

There are16 independent worlds and1536 original/renamed query pairs. The bound
uses complete29-way query distributions. It does not establish an all-position
KL failure by itself: its query-only contributions to the51-position average
are0.000282/0.000125, below0.001. Other positions were not included in this audit.

Both-correct pairs still contribute0.003881 IID and0.004801 OOD to the respective
population-mean floors. Thus the obstruction does not come solely from a few
wrong answers. The remaining contributions are additive sums over disjoint
sample categories, not explained-variance fractions:

* IID:16 correctness-changing pairs contribute0.010079; one pair with two
  different wrong answers contributes0.000430.
* OOD:3 correctness-changing pairs contribute0.001590.

The next discovery corpus contains all20 answer-disagreement pairs as well as
all1536 scored pairs. It can support causal explanations of the native model's
identity-sensitive behavior; it must not be filtered out of fidelity evaluation.
No candidate intervention has yet explained those failures.

## Receipts and correction

[Lower-bound receipt](../ENTITY_EQUIVARIANCE_LOWER_BOUND_V1.json) passes six controls:
the KL-mixture identity, attainment per pair, identical distributions, correct
vocabulary alignment, a live inverse-alignment negative, and the log2 limit.
[Complete case corpus](../ENTITY_SYMMETRY_COUNTEREXAMPLES_V1.json) retains every pair.

The initial output used a misleading `can_meet` label for small lower bounds.
Its unchanged bytes are preserved in
`ENTITY_EQUIVARIANCE_LOWER_BOUND_V1_INITIAL_LABELS.json`; the canonical output says
`not_ruled_out_by_pairwise_bound`. Numerical results did not change. The proof
does not justify a full-model impossibility claim, a smaller-program success, or
relaxing the user's requirement to predict native distributions and interventions.
