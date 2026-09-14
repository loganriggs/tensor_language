# Spherical-mean carrier correction

September14 10:44UTC. Explicit affine Down_bias carrier failed to improve
native error. Test omitted even-product mean, without fitting any native states.

For a unit-RMS isotropic spherical MLP input, E[xx^T]=I and therefore
E[(Lx)elementwise(Rx)]=(L elementwise R)1. Compute this from weights for
each MLP0..16, add its stored bias, and combine through residual carry
coefficients to the input of block17. This assumes a separate isotropic
MLP-input law, zero attention/initial mean; it is NOT an expectation under
the actual recursively generated model or a claim of native isotropy.

Use this b to fold K=(T-That)(b,.) and execute original sparse+Ka.
Same1536FP32extra values/MACs as prior carrier. Freeze before native scoring.
A algebraic recurrence and FP32correction-at-b replay<=1e-6.
B serialized bytes<=1%above original. C >=5%raw12-output error-energy
improvement on both historical120 and fresh96 panels. No data fitting,
adaptation, new native forwards or adoption claim. CPU120s/two threads.
