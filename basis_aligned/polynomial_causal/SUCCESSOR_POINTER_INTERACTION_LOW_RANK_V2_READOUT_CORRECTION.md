# Successor pointer interaction low-rank V2 readout correction

V1 stopped invalid before HOLDOUT because direct readout of a cached late-native query residual differed from the native model logits by `2.09808349609375e-5`, above the preregistered `1e-5` exactness bar. The helper had converted the query residual to FP32 and evaluated a query-only tensor, whereas the native forward evaluated the full residual tensor in its native dtype. This was an instrument mismatch. No length-seven model outcome was opened.

V2 preserves rows, FIT/HOLDOUT separation, SVD ranks, selection rule, causal arms, metrics, bars, and price. It clones the recipient's full cached final residual tensor, replaces only the query row with the candidate state in the cached native dtype, and then applies the same full-tensor RMS, unembedding, and tanh softcap before selecting the query logits. The native replay arm uses the same path. V1 rank outcomes remain scientifically uninterpreted.
