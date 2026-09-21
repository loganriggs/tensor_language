**Exact root identities are absent in the tested model-derived banks**

The prediction that at least one tested bank would have dependent root products fails. Both random probe seeds give full column rank:

| Quadratic bank | Root products | Observed rank | Smallest / largest evaluation singular value |
|---|---:|---:|---:|
| Earlier four-feature learned bank |10|10|0.0191–0.0767|
| Compressed four-feature learned bank |10|10|0.00781–0.0131|
| Six original source reads,1152inputs |21|21|0.00191–0.00281|

The first two are exact span coordinates for their entire learned full-input feature functions, not amplitude restrictions. They remain replacement-bank approximations to the original teacher. The third uses the actual six original source quadratic forms. Duplicate-feature and linear-combination controls correctly return deficient rank6out of10, while the independent three-feature control has rank6out of6.

Any exact polynomial dependency must vanish at every evaluation point. Thus full column rank witnesses its absence in exact arithmetic; these float64 singular values supply numerical evidence well above the1e-10cutoff. Evaluation condition numbers do not certify full coefficient-norm conditioning or OOD fidelity.

**Exact coefficient-metric successor**

For symmetric quadratic forms A,B,C,D, the inner product of their symmetrized quartic products is

$$
\left\langle\operatorname{Sym}(A\otimes B),\operatorname{Sym}(C\otimes D)\right\rangle
=\frac{\operatorname{tr}(AC)\operatorname{tr}(BD)+\operatorname{tr}(AD)\operatorname{tr}(BC)+4\operatorname{tr}(ACBD)}{6}.
$$

The implementation agrees with explicit24-permutation dense symmetrization to3.13e-16relative error. Applying it to the two compact banks gives full rank10in both cases. After normalizing individual product coefficient norms, minimum/maximum Gram eigenvalue ratios are0.02425and0.0005049. The latter has appreciable approximate dependence, but no exact nullspace; its numerical rank is not a sparsity or causal result. Conditioning depends on the saved covariance-scaled input geometry.

**Research consequence**

Do not apply the five-input affine-nullspace algorithm unchanged inside these frozen dictionaries: the freedom it exploits is absent. The earlier restricted-target improvement does not demonstrate a free exact rewrite of these full-input bank products. Approximate refactoring and changing or enlarging the quadratic dictionary remain possible, and previous root-function refits already explored some of that space. Avoid relabeling that prior work as a new exact-identity method.

For the full decomposition objective, the useful retained tool is an explicit dependency test before searching alternative polynomial representatives. A genuinely new structural move must change the dictionary or exploit relations in another native factorization, while paying its leaf and shared-product costs. No circuit is adopted, and full-model extraction, OOD prediction, selective removal and stable reuse remain unproved.

[Plan](NATIVE_BANK_DEPENDENCIES_PLAN_V1.md) · [Evaluation rank results](NATIVE_BANK_DEPENDENCIES_V1.json) · [Exact coefficient Gram](NATIVE_BANK_EXACT_GRAM_V1.json) · [Contraction implementation](audit_native_bank_gram.py).
