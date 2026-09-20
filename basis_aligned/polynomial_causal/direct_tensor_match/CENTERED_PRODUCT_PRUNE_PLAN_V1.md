# Native centered product deletion — 2026-09-20 19:48 UTC

Freeze the training-selected centered_floor01 rank8linear/width8quadratic Muonseed1 program. Enumerate all256subsets of its8quadratic products. Refit output coefficients by exact Gaussian quadratic Gram/cross contractions; adjust the constant to preserve the frozen program's Gaussian mean. Keep center and linear program fixed. This optimizes removal of products from this fixed dictionary, not arbitrary feature directions or native fullteacher fit.

Select one subset at eachwidth0..8 by smallest weighted coefficient error; equivalent toGaussian functional error withfreeconstant. Use pseudoinverse rcond1e-12 and report normal-equation residuals and rank. Total scalar price20,736+3,456k;8products48,384,4products34,560. Compare true empiricalfullquartic errors only afterselection; no panelsselectsubsets.

pred_a: full8product replay<1e-8; normal-equation residual<1e-8; minimal loss nonincreasing withwidth (tolerance1e-10 relative); exactGaussianmean preserved<1e-10.
pred_b: best4product subset retains>=95% of frozenquadratic coefficientenergy.
pred_c: best4product panel2relativeerror increases by<.03 from frozen centered candidate.

Report all widths, centered-variation errors, costs, and all subset objective values. Null: no useful reduction within this fixed dictionary. No semantics/OOD/fullmodelclaim.
