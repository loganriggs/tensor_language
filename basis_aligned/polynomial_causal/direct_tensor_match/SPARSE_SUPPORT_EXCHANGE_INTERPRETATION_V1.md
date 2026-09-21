**Learned interaction support improves the exact coefficient objective slightly, but does not recover a faithful shared computation.**

Two fixed144-by4quadratic banks,16accepted one-for-one product exchanges each, from the predeclared256newpair pool. All16readoutcoefficients perproduct are refitted after each edit. Runtime9.55seconds. Integrity PASS; registered>=10%capturedscore improvement FAIL; component10%gate FAIL.

|Metric|Seed1101 before→after|Seed1102 before→after|
|---|---:|---:|
|Captured regularized coefficient score|+1.524%|+1.204%|
|Sampled coefficient error|96.916→96.826%|97.215→97.197%|
|StandardGaussian function error|91.501→91.477%|91.555→91.572%|
|Opened text error|65.657→65.669%|63.821→64.088%|
|Root1 same-token response error|51.816→51.144%|49.558→50.067%|
|Root1 sensitivity error|108.380→108.111%|107.876→107.737%|

The objective improves monotonically by construction, but that does not guarantee the sampled diagnostics or data-weighted behavior improve. Both observed coefficient-query errors decrease slightly; neither dictionary approaches good absolute fidelity. Text slightly worsens. Do not equate gain in omitted-constant capturedscore with equal percentage reduction in total tensor error.

This is an actual discrete graph edit:512selected rootproducts now include16newpairs perstart,496remain. It goes beyond the old seeded support. It is not arbitrary-DAG search: the144quadratic producers, their four input products, diagonalrootpairs and768candidate pool are fixed. Only16greedy exchanges were allowed. Thus the null does not certify the optimal sparse support, the best hierarchy or the impossibility of a simpler circuit.

**Executed CPU graph audit:** frozen producerfactors andwriter match sourceartifacts exactly; all144quadratics remain live;512selectedpairs are unique; every diagonal is retained. Direct execution that recomputes each root separately agrees with shared execution within2.4e-15. Final cost remains1088variableproducts,1353728floatcoefficients,1024indices. Recomputing quadratic producers separately for each root would use4032products (while still computing each square's producer once); this existing sharing benefit is not a new saving from the edits. Consumer counts range1–15and1–17. Dense linear coefficient work is accounted through storage, not silently included in variable-product count.

The higher-cost mixed-objective CP candidates remain the stronger conditional/finite-removal baseline. Their improvement came from changing the metric and feature directions, while this edit test optimized coefficients under the original global metric. Therefore this comparison does not establish CP superiority over a shared hierarchy with the same objective. The next useful bridge is exact mean/covariance matching for the shared quadratic-product dictionary, followed by the same component and finite-removal checks. Avoid expanding512products-of-quadratics into8192independent CP atoms for itsGram; shared quadratic moments can preserve the reuse computationally.

No new semantic unit, OOD prediction, extraction or composition result is claimed. Earlier capitalizationselectivity and globalcoefficients remain unresolved.

[Native results](SPARSE_SUPPORT_EXCHANGE_NATIVE_V1.json) · [Preregistered plan](SPARSE_SUPPORT_EXCHANGE_NATIVE_PLAN_V1.md) · [Graph audit](SPARSE_SUPPORT_EXCHANGE_GRAPH_AUDIT_V1.json) · [Higher-cost finite-removal baseline](MIXED_CP_REMOVAL_INTERPRETATION_V1.md).
