# Fresh transfer: selectivity passes, quadratic prediction fails

The frozen48text panel has100%native number capability in every cell.
Original five sources, full six sources and the substituted five sources each
pass all16native selectivity cells. Thus the source substitution has **no
selectivity advantage on this panel**. Its fresh quadratic prediction fails.

| Method | Native selective cells | Worst quadratic number error |
|---|---:|---:|
| Original5 |16/16|7.773%|
| Full6 |16/16|17.987%|
| Substituted5 |16/16|16.751%|

The two prediction failures occur in the same cell: opposite-number distractor,
subject edit, beside_subject singular. All modal-prediction errors remain
within the5%number-budget gate. Native collateral remains below8.2%.
Source support and solver rules were frozen before outcomes; all sources use
their own gradient/Hessian subblocks and exact-null initialization. Original5
therefore remains the passing matched-width prediction baseline here.

The native receipt passes exact closure, numerical replay, directional
finite-difference and count gates. Costs:12prefix,72double and40native suffix
calls,32gradient and192Hessian-row reverse calls. The most advanced candidate
still requires native source/derivative generation; there is no extracted
standalone circuit or general unrelated-behavior preservation claim.

## Opened-data negative-result redteam

Freeze the three selected amplitude vectors and evaluate radii1,0.5,0.25.
For each direction a compute analytic D³F(0)[a,a,a] using one scalar parameter
per input. Compare Q(t)=-tG.a-0.5t²a^THa with
C(t)=Q(t)-t³D³F(0)[a,a,a]/6. This is a directional diagnostic, not a stored
full cubic coefficient tensor, shared-feature discovery or prospective repair.

| Method | Half-radius quadratic max error | Full-radius cubic max error |
|---|---:|---:|
| Original5 |2.802%|3.416%|
| Full6 |3.326%|7.208%|
| Substituted5 |3.296%|6.385%|

All full-radius cubic predictions pass10%number/5%modal. But the registered
claim that cubic error improves every cell fails:7of48full-radius cells worsen.
The half-radius remedy also changes intervention strength, so it cannot inherit
the full-radius selectivity result. Neither diagnostic overwrites the original
fresh quadratic failure.

Independent directional first/second derivatives match the multivariate
projections within7.99e-15. Full-radius native outcomes replay exactly. Third
derivative finite differences pass the predeclared15%relative-or1e-7absolute
gate (worst relative13.953%atstep0.05). Native/double differences<=7.44e-6.
Counts:12prefix,208double,88native calls,96first+96second+96third reverse calls.
Thus higher-order local response is supported as the missing approximation;
the tests do not suggest an axes/sign or derivative-computation bug.

## Concrete next path

The final readout has shared denominator
q=mean(h²)+eps and eight token numerators n_v=U_v.h for four contrasts:
F_o=30[tanh(n_left/(30sqrt(q)))-tanh(n_right/(30sqrt(q)))].
Keep this normalization and softcap explicit. Approximate the nine upstream
fields along the same frozen direction to second order, then evaluate the
exact readout. It shares one denominator across outputs and reuses t,t².
This tests whether final-readout nonlinearities account for the failed
quadratic approximation without requesting native third derivatives.
Count27per-ray field coefficients versus16for the cubic output jet, plus
direction storage and all native generators; no automatic simplicity win.

Receipts: `source_ood_v2_result.json` and
`source_ood_v2_cubic_radius_result.json` in the followup directory.
