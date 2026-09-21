**Return to full joint quadratic tensor capacity**

Target T_vij=sum_k C_vk sym(A_ki B_kj), C=UD17,A=L17 E,B=R17 E,E=[I,lambda17 D16,O17]. Exactly contract output and either symmetric input unfolding Grams. Use Cholesky of U^T U and EE^T to preserve ambient Euclidean metrics without storing huge QR frames. Normalize factors by global scalars only; relative spectra unchanged. No activation fitting.

Dense small-tensor controls must agree<1e-12. Native two-mode trace discrepancy<1e-9 and minimum relative eigenvalue>=-1e-10. Predict output-rank128and shared-input-rank512 necessary relative Frobenius error floors each<20%. Report ranks16through1152 and necessary ranks for50/20/10/5% error regardless. These are individual unfolding lower bounds, not achievable joint Tucker errors. They do not rule out arithmetic programs with large linear spans and sparse internal computation.

Scope is full original folded quadratic numerator, unlike the recent six-read and five-amplitude diagnostics. RMSNorm, biases and attention source dependence remain outside this target. This is a structural feasibility baseline for future whole-target fitting, not a circuit identification result. Zero model forwards; managed GPU contractions only.
