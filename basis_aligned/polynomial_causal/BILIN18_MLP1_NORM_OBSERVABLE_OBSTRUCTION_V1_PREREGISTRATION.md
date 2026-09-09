# Exact obstruction to a shared linear-coordinate plus norm quotient

Follow-up to the valid full-input MLP1 joint-reader weight audit. That audit
found full numerical input support and distinct task function spans. Its support
calculation excludes a smaller linear representation of the numerators alone.
It does NOT by itself exclude a representation using the already retained norm:
Q=I is a counterexample, since x^T Q x=||x||² needs no linear coordinates.

The actual question is whether all eight normalized reader outputs can be
computed from (U^T x, rho=||x||²) for some proper linear subspace U, for EVERY
real input x. Arbitrary decoding from this state is allowed. Bias and native
positive epsilon are explicit, so this is equivalent to preserving all quadratic
numerators. Orthogonal rotations/reflections in the discarded space imply
Q_k=U H_k U^T + alpha_k I. Therefore every discarded vector is a common
eigenvector of every symmetric Q_k. An invertible commutator of any two reader
forms rules out this quotient, including arbitrary nonlinear decoding.

Fix exactly the first serialized temporal reader and first iswas reader from
BILIN18_MLP1_JOINT_READER_WEIGHT_V1_READERS.json. No pair/rank/module search.
Treat saved FP32 readers and trained FP32 Left/Right/Down weights as exact dyadic
rationals. Use prime65521 and reduce every dyadic coefficient into its finite
field. Compile Q symmetrically via exact-integer FP64 GEMMs: each dot product is
a nonnegative sum below2^53, and each multiplicand/product is an exact integer.
Take K=Q_temporal Q_iswas - Q_iswas Q_temporal modulo65521, then Gaussian
elimination modulo this prime. A nonzero determinant proves rational K is
invertible and hence the stated proper linear-plus-norm quotient is impossible.
A zero determinant is inconclusive, not evidence for a reduction.

A: frozen source/reader/parent/checkpoint hashes; small rational-vs-modular
contraction and determinant controls; integer/finite/remainder and dot-bound
tripwires; symmetry/antisymmetry identities; saved matrix reload; independent
CPU integer determinant replay equals the GPU determinant. No thresholded rank
is used as the certificate. Matrix shape1152x1152, actual reader selection fixed.
B: exact modular determinant is nonzero. C: real-arithmetic scope and full native
price recorded. C is descriptive metadata, not an additional scientific bar.

Use the existing backend loader without forwards, no fit/update/training,
managed GPU only, alarm600s. Persist the modular commutator in a compressed
array artifact for independent CPU replay; source hashes bind its provenance.
All native weights and adapters remain charged. This is a bounded mathematical
obstruction for one representation class, NOT a proof that circuits cannot
exist, nor a restriction on a smaller nonlinear arithmetic program, domain-
restricted circuit, or the already derived exact edit-response state.

If certified, close this full-domain linear-plus-norm state family at MLP1 and
use explicit nonlinear producer/factor operations for further circuit work.
If inconclusive, preserve the failed certificate attempt without scanning pairs.
