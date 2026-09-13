# Sparse interaction regional validation

Frozen choices: output-only and output/head HOSVD frames; coefficient error budgets 2%, 5%, 10%; largest-energy support, fitted only from weights. Score every choice, without selecting on text. Use the existing 120 historical compact predictor examples in five groups.

Replace only the mixed numerator T(zbar,a) in the selected twelve output logits. Recover head coordinates a by least squares from compact minus MLP-only linear states; report reconstruction residual due to native rounding. Keep linear, head/head, background, normalization and saturation functions supplied by the reference. This tests conditional mixed-operator approximation, not end-to-end extraction or net model savings.

A: exact T contraction versus direct L/R mixed formula relative error <=1e-10; cache margin replay <=1e-4 absolute; recovered write relative residual <=1e-3 aggregate. B: each candidate has <=5% additional compact target-effect error in every group. C: each candidate preserves every reference sign with magnitude >=1e-5. Retain failures and all signed outputs. Untargeted control readers are unchanged by construction and are not evidence of learned selectivity. No new body forwards or data fitting.
