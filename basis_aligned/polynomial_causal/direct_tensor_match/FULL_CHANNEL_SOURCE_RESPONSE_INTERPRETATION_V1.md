**The source-intervention gap is already present in the local derivative**

The analytic normalized endpoint derivative passes five independent small autograd controls (worst1.01e-15), native central differences (2.75e-9), and replay of the preceding full-size swap errors (5.46e-8). Managed runtime8.953s. Both scientific predictions fail: the tangent does not meet10%everywhere, and large swaps do not universally amplify error by25%.

| Cohort | Infinitesimal logit-response error | Full swap error | Common-native-background tangent error |
|---|---:|---:|---:|
| FineWeb continuation |13.63%|14.77%|12.88%|
| FineWeb spaced word |13.16%|12.27%|13.15%|
| Code continuation |12.39%|13.78%|10.41%|
| Code spaced word |9.26%|10.49%|8.52%|

Responses include the derivative of RMS17, the bilinear layer or affine-corrected replacement, residual identity, final RMS and logit softcap. The common-background column uses the native output-state Jacobian for both residual tangents; it retains three failures. Residual-space tangent errors are only4.68–9.17%, showing that the downstream output geometry matters. Comparing these relative errors does not constitute an additive causal attribution of the discrepancy.

This rejects a large-displacement-only explanation. It does not establish that a particular derivative loss will transfer, or that input normalization is irrelevant. The next conditional baseline explicitly includes source-response matching, evaluated separately from coefficient matching. It keeps the3686products and prices the affine repair, so gains cannot come from added capacity.

The successor uses2048historical states, not the opened48-document evaluation panel. For each state it centers the original MLP16 source write, propagates that direction through recipient RMS17, and fits the affine-repaired quadratic's directional response. The centered product dictionary has derivative features

$$
g_k(x,v)=(a_k^\top(x-\mu))(b_k^\top v)
+(a_k^\top v)(b_k^\top(x-\mu)).
$$

Its empirical response Gram adds to the covariance coefficient Gram. Each term is normalized by its teacher energy; lambda0,.1,1,10are recorded, with lambda1the fixed primary. Conditional output weights have an exact linear solve. Five independent direct least-squares/autograd controls pass below1e-15. This objective measures the unembedding-weighted polynomial response before final normalization, so endpoint improvement remains a test rather than an implication.

[Response results](FULL_CHANNEL_SOURCE_RESPONSE_V1.json) · [Analytic implementation](normalized_bilinear_response.py) · [Response-feature controls](centered_product_response.py).
