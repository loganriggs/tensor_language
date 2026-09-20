# Normalization-aware artificial probe comparison

Input audit: actual squared radius is 1152 with relative standard deviation
9e-8; calibration Gaussian predicts relative standard deviation .1101.
Known-identity control sphere_metric_control.py shows an exact degree4-to-degree2
simplification on the sphere can have positive Gaussian error. This does not
show that the native target contains a large such component.

Next native diagnostic: freeze the centered Gaussian-linear four-product
program and mean-corrected quartic rank8 candidate. Draw paired artificial
samples z=mu+chol(M)epsilon and xn=sqrt(1152)*z/||z|| using calibration moments
only. Evaluate native folded weights and both programs on each, with 2048
samples per law, separate fixed seeds 2033/2034. Report functional errors,
mean/variation split, and candidate ordering against previously reused text
panels. Normalization changes moments; report measured changes, do not call
this the same Gaussian or covariance. No fitting in this first comparison.
Predictions: normalized-law errors closer to text-panel errors for both
candidates than Gaussian-law errors; quartic beats quadratic on both normalized
probe seeds; teacher and saved program replay remain finite and FP32/64
relative disagreement <1e-3. Preserve failures. Native runner pending.
If the law comparison supports it, follow with equal-budget artificial-probe
optimization on raw and normalized samples with fresh probe validation. This
would remain weights-first discovery with declared data-informed input moments.
