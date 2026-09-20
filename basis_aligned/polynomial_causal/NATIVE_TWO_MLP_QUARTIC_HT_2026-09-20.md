# Native two-MLP quartic fold and pair-tree HT baseline

This benchmark folds a specific native-weight path, rather than another fit to the same quadratic response dictionary. At a nominated noun position, K in R^(1152x5) contains the five source sensitivities at the input of MLP11, and q in R^(4x1152) contains four native downstream readers after MLP12. Let B_l(x)=D_l[(L_l x)*(R_l x)] denote the homogeneous numerator only. The benchmark object is

    f(a) = q B_12(lambda_12 B_11(K a)).

It is homogeneous degree4 in five source amplitudes. S[d,i,j]=sum_h D11[d,h] sym((L11 K)[h,i](R11 K)[h,j]); fold lambda12 into S. Then

    H[o,i,j,k,l] = sum_h (q D12)[o,h] (L12 S)[h,i,j] (R12 S)[h,k,l].

This contracts the native widths without constructing a fourth-order residual-width tensor. Each context gives four5x5x5x5 tensors,2500 raw coefficients. It is one direct two-MLP numerator branch. It excludes background/bias terms, RMS denominators and the intervening attention path. Native derivative frames/readers remain context dependent and fully charged. This is not a fixed polynomial representation of the full normalized network, a finite-edit predictor, or a causal circuit.

## Native replay and HT comparison

The exact fold passes64 direct polynomial evaluations across16 context groups,96 opened texts and two sites: relative error6.59e-15. Full pair-tree reconstruction passes relative coefficient error4.00e-15. Native source capture costs24prefix,16suffixforward,80sourceJVP and64readerreverse calls. The initial run failed at an export helper invocation; its runner/log remain. The corrected run changes only export arguments and output paths.

The pair tree keeps five-dimensional identity leaves and the four-dimensional output mode, and factors pair modes(ij) and(kl) via their SVDs:

    u_alpha(a)=sum_ij A[i,j,alpha] a_i a_j
    v_beta(a)=sum_kl B[k,l,beta] a_k a_l
    f_o(a)=sum_alpha,beta C[o,alpha,beta] u_alpha v_beta.

This is a restricted HT baseline with dense transfer cores; no sparse-core learning or cross-branch feature merging is claimed. Source: [Grasedyck, Hierarchical Singular Value Decomposition](https://www.mis.mpg.de/publications/preprint-repository/article/2009/issue-27). Literature search20September queried "Grasedyck hierarchical singular value decomposition tensors 2010 hierarchical ranks matricization" and opened the primary institute abstract. The mapping here is to the finite quartic coefficient tensor with fixed identity leaves, not to the nonlinear model. The paper's compression framework supplies no causal or feature-identification guarantee for this model.

The tested error is ||Sym(H-Hhat)||_F / ||Sym(H)||_F, so two representatives of the same repeated-input polynomial are compared functionally at coefficient level. This is not empirical native-behavior error.

| Representation | Values/context | Worst polynomial coefficient error |
|---|---:|---:|
|Raw tensor|2500|Exact|
|Canonical degree4 monomials|280|Exact|
|Rank8 tree, native pair representative|656|12.229%|
|Rank8 tree, symmetric representative|656|9.352%|
|Rank2 tree, native pair representative|116|36.086%|
|Rank2 tree, symmetric representative|116|36.257%|

Canonical polynomial price is4*C(8,4)=280 coefficients. Shared monomial index ordering is fixed algorithmically. Its independently grouped coefficient evaluation replays the native folded polynomial to3.02e-15. Tree storage is2*25*r+4*r*r, with identity leaves implicit. Thus the rank8 symmetric tree passes the10% approximation bar but loses to the exact canonical polynomial on stored values. Rank2 saves values but fails fidelity. Compute and native-producer costs are not proven lower by either tree; there is no simpler-program adoption.

The planted control (a^T a)^2 shows raw pair rank1 versus fully symmetric pair rank15 at dimension5, with identical polynomial replay3.56e-15. On the native data, however, the symmetric rank8 fit has better polynomial error. Therefore symmetry can increase representational rank yet still help a particular truncated approximation. Neither representation should be chosen solely from the toy example.

## Evidence and limits

Native result/artifact: ../bilinear_quotient/circuits/followups/native_two_mlp_quartic_ht_v1r1_result.json and native_two_mlp_quartic_ht_v1r1.pt. CPU price/control receipt: NATIVE_QUARTIC_HT_CPU_AUDIT.json. Implementations: quartic_pair_tree.py, audit_native_quartic_ht.py and managed run_native_two_mlp_quartic_ht_v1r1.py.

The random-amplitude CPU check also fails rank2 strongly (60.2%/89.8% for native/symmetric representatives). It is algebraic validation, not text OOD. Native weights alone do not establish that this isolated numerator carries the behavioral effect. This fills one actual folded-weight HT baseline gap, while full normalization, causal validation, adaptive trees, sparse transfer cores and reusable features remain unresolved.
