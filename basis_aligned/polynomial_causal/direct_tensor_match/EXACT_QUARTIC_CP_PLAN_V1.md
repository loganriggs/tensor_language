# Exact-gradient quartic CP route — 2026-09-20 16:42 UTC

Native sampled-entry quartic fits failed. A separate exactly rank-one planted quartic shows that accurate scalar norm estimates can coexist with unusably noisy gradients: at d1152,256queries, relative gradient RMSE exceeded4million for the tested random orientation. This motivates exact gradient contractions, not larger sample counts by default. This toy is not a measurement of native gradient SNR.

Student family: sum of output-weighted products of four linear forms (quartic CP with full input symmetrization). It differs from the shared quadratic DAG and does not automatically share quadratic subcomputations. Price5*d*r in reduced equal input/output dimensions, plus frame accounting.

The student self inner product is the sum over24 input permutations of products of four pairwise factor Gram matrices, weighted by the output Gram. The teacher–student cross inner product is a weighted sum of exact teacher directional contractions along each student term's four vectors. Each directional contraction is available from the two native bilinear layers using the same three-pairing formula as the coefficient-entry oracle.

Thus ||student||²−2<teacher,student> and its gradient can be computed exactly without expanding H or sampling coefficient entries. The omitted teacher norm is constant in student parameters; a separately estimated norm affects reported normalized errors, not the exactness of the unnormalized parameter gradient. Avoid claiming exact relative error from an estimated denominator.

Implementation `quartic_cp.py` and dense24-permutation value/gradient checks are complete (<1.3e-15). Next: small planted CP optimization controls, including output-weight elimination and parameter scaling, then a bounded native comparison. Keep CP versus shared-DAG capacity differences explicit. The old finite-query output-rank128 lower bound78.44% is a relaxation; it does not prove that the full shared-DAG family can attain that error.
