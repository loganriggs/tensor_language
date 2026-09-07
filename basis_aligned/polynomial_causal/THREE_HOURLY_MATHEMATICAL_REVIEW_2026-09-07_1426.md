# Three-hour mathematical tensor-network review — 2026-09-07 14:26 UTC

## Decision

The five-source rank-16 program has now reached the right object for the user's proposed weight translation: not a set of privileged hidden neurons, but paired task-conditioned subspaces inside each exact native MLP weight map. The next circuit split should use the canonical angles between temporal and is/was source-response subspaces, pull their shared and contrast directions through `Down^T Q`, and causally patch those directions in the complete model. This is a cross-boundary grouping test: a shared coordinate may be written by different hidden units or different MLPs, while one native MLP may split into shared and task-contrast pieces.

The complete five-site Boolean lattice supplies an unusually strong simplifying premise. Its maximum degree-at-least-two Mobius energy is `.00138`, so site-level support is almost additive on the selector population. This licenses a prospective greedy composition of task-mode groups after singleton full-patch validation; it does not license replacing causal patches with weight scores.

## Explicit local tensor object

For source MLP `s`, residual width is `d=1152`, hidden width is `h=4608`, and the frozen source basis has rank `q=16`. Write the bilinear MLP as

\[
y_s(x)=D_s\big[(L_sx)\odot(R_sx)\big],
\quad L_s,R_s\in\mathbb R^{h\times d},\quad D_s\in\mathbb R^{d\times h}.
\]

Let `Q_s in R^{d x q}` be the orthonormal causal source basis and

\[
A_s=D_s^{\mathsf T}Q_s\in\mathbb R^{h\times q}.
\]

This is the exact compiler already verified at relative error at most `8.11e-14`. For task `t` and fitted donor/base pairs, let

\[
\Delta H_{s,t}\in\mathbb R^{n_t\times h},\qquad
Z_{s,t}=\Delta H_{s,t}A_s\in\mathbb R^{n_t\times q}.
\]

`Delta H` contains changes in the elementwise bilinear hidden products. The top right singular subspace
`U_{s,t} in R^{q x r}` of `Z_{s,t}` is therefore a task-conditioned *write subspace* expressed inside the causal `Q_s` coordinates. The current empirical rank boundary motivates `r=4`; rank remains a matched-capacity control, not the identification claim.

If

\[
U_{s,T}^{\mathsf T}U_{s,I}=P_s\,\mathrm{diag}(c_{s,j})\,V_s^{\mathsf T},
\]

the canonical vectors are `a_j=U_T P_j` and `b_j=U_I V_j`. For every nondegenerate pair define

\[
m_j={a_j+b_j\over\sqrt{2+2c_j}},\qquad
k_j={a_j-b_j\over\sqrt{2-2c_j}}.
\]

The `m_j` are shared/mean directions and `k_j` are task-contrast directions. The exact physical hidden writer coefficients are `A_s m_j` and `A_s k_j`; the physical residual directions are `Q_s m_j` and `Q_s k_j`. For a direction matrix `B`, the intervention installed at the MLP output is

\[
\Delta y_s^{(B)}=(\Delta H_s A_s B)(Q_sB)^{\mathsf T}.
\]

This is executable without treating hidden-unit identity as canonical.

## Gauge, symmetry, polynomial degree, and price

Under the legal orthogonal gauge `Q_s -> Q_s G`, `A_s -> A_s G`, and task coordinates `U_{s,t} -> G^T U_{s,t}`. Both `A_s B` and `Q_s B` are invariant physical objects after the induced coordinate change. Canonical pairs retain sign, ordering, and rotations within exactly degenerate singular-value blocks; individual vectors are not identifiable inside a degenerate block. The new helper tests this gauge covariance directly and passes 3/3 tests.

Locally, `Delta y_s^(B)` is degree two in the residual input because the hidden features are `(Lx) odot (Rx)`. The complete transformer remains nonlinear and nonpolynomial because later normalization and attention operations intervene; therefore local weight exactness does not imply global additivity. Allowed inputs are the registered paired text populations. Outputs to preserve are task-mode response vectors, answer-margin effects, and full-vocabulary/control behavior. Approximation is measured by relative squared response error plus signed causal projection, not activation variance.

For `k` retained directions, the literal local interface stores or derives `A_sB` (`h*k`) and `Q_sB` (`d*k`) values and contracts roughly `O(nhk+ndk)` operations. Since both are derivable from native weights and the frozen basis, this is currently an interpretation/execution price, not yet a model-storage saving. Adoption still requires end-to-end priced comparison.

## Theorem and algorithm mapping

Bjorck and Golub's principal-angle construction computes the canonical correlations and vectors of two subspaces from an SVD of their cross-Gram matrix ([Numerical Methods for Computing Angles Between Linear Subspaces](https://rainbow.ldeo.columbia.edu/~alexeyk/BjoerckGolub1973.pdf)). The canonical CS representation makes the pair-of-subspaces geometry explicit and supplies backward-stable algorithms, including minimal common and orthogonal pieces ([A Canonical CS Representation of a Pair of Subspaces](https://epubs.siam.org/doi/10.1137/15M1009573)). These results map exactly to `U_T^T U_I`: the singular values are the cosines `c_j`, and the sum/difference vectors above are the two orthogonal axes in each canonical plane.

Wedin's singular-subspace perturbation theorem bounds rotation by perturbation size divided by a singular-value separation ([Perturbation Bounds in Connection with Singular Value Decomposition](https://doi.org/10.1007/BF01932678)). Its executable consequence here is a gap audit: a source mode is not stably identified merely because two fitted bases have high cosine; the retained singular block must be separated from the discarded block, and even/odd-document estimates must obey the observed principal-angle bound. Degenerate blocks are reported as blocks, never named coordinates.

JIVE decomposes multi-block matrices into joint, individual, and residual low-rank structure under orthogonality conditions that make its joint/individual components identifiable ([Joint and Individual Variation Explained](https://pmc.ncbi.nlm.nih.gov/articles/PMC3671601/)). It is a useful neighboring formulation but does not directly solve this object: temporal and is/was rows are not paired observations of the same samples; the causal output metric is not unweighted Frobenius variance; and downstream execution is nonlinear. Consequently, JIVE can motivate a shared/specific decomposition control, but the CS/principal-angle object plus held-out causal intervention is the correct exact restriction.

No cited theorem identifies a neural circuit from static weights. Theorems identify subspace geometry and stability conditional on the empirical `Z` matrices. Full causal patches are still the membership test.

## New empirical facts affecting the object

The frozen mask30 program—top50 hidden support at MLP0, full support at MLP1/2/3/6—transfers to temporal-v13/is-was-v12 and is indistinguishable from the full parent on target behavior and response. Its two control flips are exactly the same rows flipped by the full parent. Mask30/full centered control logits differ by only `1.87%` of the full-parent effect, so the MLP0 split is stable; the full parent's control scope is the remaining limitation.

The just-finished DAS tournament is formally invalid because its authority check required the label `pooled`, while the predecessor records the same step-zero source as `deterministic_pooled`. Raw scientific measurements are quarantined pending a zero-forward atomic audit. They already show why a single aggregate statement about regularization is unsafe: KL wins the worst-fold score (`.6968` versus no-reg `.7261`) but loses the opposite fold (`.4007` versus `.3486`). The cross-fold KL axes nevertheless have cosine `.883`, and the refit improves sealed-v12 mean/worst secondary loss from `.9803/1.0095` to `.3979/.4803` while remaining target-feasible. The preregistered requirement that regularization win both folds fails, even if the authority repair validates everything else.

## Executable consequence and opposing predictions

The next task-mode weight screen should:

1. fit temporal and is/was rank-four `U_{s,t}` independently on even and odd documents at all five source MLPs;
2. report retained/discarded singular gaps, canonical cosines, projector stability, and gauge-safe physical `A_s m_j` / `A_s k_j` participation;
3. freeze shared-mean and task-contrast groups before opening the construction holdout;
4. full-patch each group alone, its complement, and the greedy union, measuring vector response, behavior, and matched controls.

Opposing outcomes are explicit. Stable mean axes that causally move both tasks identify shared writers; stable contrast axes with opposite or task-selective effects split a native MLP; unstable axes or weight-ranked patches with no causal effect demote the weight object to an incidence prior; failure of the mean+contrast union to replay the rank-16 effect invalidates the instrument. Because the site-level lattice is nearly additive, a greedy union is licensed only after these singleton semantics hold on held-out rows.

This route dominates another hidden-percentile sweep: it can change computational specification, within-module splitting, and cross-task reuse. The DAS tournament remains complementary; its corrected audit will determine whether KL provides robust worst-family selection, but it cannot replace the source-to-weight causal patches.

Next mathematical review due around **2026-09-07 17:26 UTC**.
