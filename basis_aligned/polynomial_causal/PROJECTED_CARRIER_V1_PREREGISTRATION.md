# Homogeneous projected-carrier correction

September14 10:47UTC. The synthetic pushforward mean's constant correction
passes raw-output tests but changes homogeneous bilinearity. Correct this
structural issue without a new fit: q=b/||b|| from the frozen pushforward
carrier, K=(T-That)(q,.), output=That(z,a)+(z.q)(K a).

In exact arithmetic the coefficient error is the old error projected away
from q on the residual axis. It vanishes at either zero input and exactly
reproduces the target on span(q). Every scalar contrast matrix spectral norm
cannot increase under this orthogonal projection. This absolute fact does
not imply native-relative improvement because errors can cancel.

A FP32executor vs decoded dense coefficient<=1e-6; both zeroport outputs0.
B serializedbytes<=1%above original. C >=5%all12normalizedrawerror improvement
on both historical120 and fresh96 panels. D each of six contrast spectral
norms<=original*(1+1e-6). Count2688extra FP32values/2688MACs+12multiplies
perrow; no native training/alpha selection. Freeze before native scoring.
CPU120s/two threads. Task-specific contrast validation still pending.
