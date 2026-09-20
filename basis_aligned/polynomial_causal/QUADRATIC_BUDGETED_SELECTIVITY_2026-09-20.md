# Curvature-aware selective edit and matched-budget control

Use the same five source coordinates and frozen analytic response phi(a)=-G a-0.5 a^T H a. Maximize signed predicted number change over[-1,1]^5 while constraining each modal effect magnitude to at most8% of the signed number effect. Native acceptance stays unchanged: at least80% aligned unitB strength and at most10% modal-to-number ratio.

Each input has five fixed starts: zero, unitB, exact-null, and5%/10%linear-budget solutions. Analytic derivatives feed SLSQP. Candidates are accepted only after direct feasibility checks; zero remains a feasible fallback. This is local nonconvex optimization, not a global certificate. Across288inputs,1440SLSQP solves plus576linear warm-start LPs are charged. No finite native outcomes choose coefficients. Twelve planted zero-curvature controls match independent8%LP optima within1.07e-14.

The first native run compares with the prior5%linear candidate; a second run uses a fair8%linear control. Each uses36prefix and120native+120reference suffix calls, with no new source derivatives. Old unitB/exact-null effects replay exactly and native/reference effects agree within5.92e-6. Both datasets are opened, so this is not prospective OOD.

| Method at8%surrogate budget | Original32 joint passes | New-construction16 joint passes | Worst native modal ratio |
|---|---:|---:|---:|
|Linear selection|29/32|12/16|10.474%|
|Quadratic selection|30/32|12/16|8.070%|

Quadratic selection gives one paired gain and no losses:42/48 versus41/48. The larger improvement over5%linear selection35/48 is therefore mostly budget choice, not demonstrated curvature benefit. Quadratic selection removes collateral failures, but six target-strength failures remain. Its median target retention is1.340/1.354 on the two datasets. Full-quadratic predictions remain within registered10%number/5%modal error bars for both candidate arms.

The overall circuit gate still fails. These are adaptive intervention procedures requiring native gradients/Hessians and source generators, not fixed shared features or an autonomous extracted model. The results motivate changing the available source interface rather than another budget sweep.

Inspection of the actual five-port capture confirms that the selected coordinates omit attention8, attention9, MLP9 and attention10 contributions. A residual-complement sixth port can restore closure of the nominated pre11 residual location while retaining the existing five ports. This would test an interface limitation directly. It would still not equal a full input edit, because other positions, x0 and first-layer value cache remain baseline; any next test must preserve that distinction and replay the old five-port subblock.

Receipts: QUADRATIC_BUDGETED_DIRECTIONS_V1.json; QUADRATIC_BUDGETED_DIRECTION_CONTROL.json; MATCHED_EIGHT_PERCENT_DIRECTIONS_V1.json; ../bilinear_quotient/circuits/followups/native_quadratic_budgeted_v1_result.json and native_quadratic_matched_budget_v1_result.json; QUADRATIC_MATCHED_BUDGET_CPU_AUDIT.json. Old gradient-budget and exact-null failures remain intact.
