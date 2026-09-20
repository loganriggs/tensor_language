# Global two-layer quartic coefficient queries — 2026-09-20 16:09 UTC

To move beyond changing five-dimensional amplitude interfaces, implement an exact query for arbitrary ordered input indices of the full two-bilinear-layer quartic. If S_ij is the first layer's symmetric residual-space quadratic coefficient and B is the second layer's symmetric bilinear map, then

H_sym[i,j,k,l] = (B(S_ij,S_kl)+B(S_ik,S_jl)+B(S_il,S_jk))/3.

Query batches need neither dense H nor a restricted input subspace. Verify values and every parameter gradient against an independently expanded tensor averaged over all24 input permutations on a small toy. Sum all entries to verify coefficient Frobenius norm.

Uniform entry sampling would give an unbiased squared Frobenius numerator, but may have prohibitive variance. Measure this on a dense two-layer toy and a single-coordinate quartic. This is instrument development and a variance falsifier before deciding whether a native stochastic full-order-five sweep is worthwhile. No Gaussian-function-metric equivalence or native success claim.
