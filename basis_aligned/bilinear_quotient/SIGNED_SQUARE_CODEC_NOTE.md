# Signed-square local normal form

The bilinear architecture admits the exact polarization identity

\[
(a^\top z)(b^\top z)
=\left(\frac{a+b}{2}^\top z\right)^2
-\left(\frac{a-b}{2}^\top z\right)^2.
\]

After the existing product codec fixes scale, sign, leg-swap, and component-order
gauges, the signed-square codec stores `u=(a+b)/2`, `v=(a-b)/2`, the shared output
factor `c`, and a graph-level subtraction. Signs of `u` and `v` are separately free
because they are squared, so their largest-magnitude coordinates are fixed positive.
The representation remains conditional: it does not solve global equivalence among
different CP decompositions.

On all five frozen native prefixes, using the same 2^-20 coefficient step shortens
the zlib-compressed canonical stream by 1.26--1.38% (34,760 to 600,872 bits). The
polarization identity is exact before quantization. Re-quantizing the transformed
factors changes the represented quadratic coefficient tensor by
2.15e-5--2.18e-5 relative Frobenius norm, measured analytically with cross-factor
Grams rather than sampled activations.

### What this says about composition

Let `f` be the product-factor program, `g` its quantized signed-square rewrap,
and `DeltaT` the symmetric coefficient tensor of the residual `e=f-g`. For two
RMS-normalized interface states `z,z'` of width `D`, polarization and
Cauchy--Schwarz give the global, factorization-invariant bound

\[
\|e(z')-e(z)\|_2
\leq 2\sqrt{D}\,\|\Delta T\|_F\,\|z'-z\|_2.
\]

The implementation computes `||DeltaT||_F` from cross-factor Gram matrices,
without materializing the `D x D x D` tensor. It also exposes the exact JVP

\[
D f(z)[h]=\sum_j c_j\left[(a_j^\top h)(b_j^\top z)
 +(a_j^\top z)(b_j^\top h)\right].
\]

The initial Frobenius fallback gives bounds 2.083, 2.771, 3.792, 4.940, and
6.445 for native prefixes k=32,64,128,256,512. That apparent capacity growth is
mostly looseness from summing error energy across output directions. Replacing
`||DeltaT||_F` by the spectral norm of its output-mode unfolding is still a
global certificate and tightens the sequence to 0.759, 0.771, 0.774, 0.780, and
0.795 (36.4% down to 12.3% of the fallback). The production result therefore
supports an approximately capacity-independent worst-case coefficient sensitivity.

There is a second, state-dependent certificate. Every homogeneous quadratic
residual satisfies the exact secant identity

\[
e(z')-e(z)=D e\!\left(\frac{z+z'}2\right)[z'-z].
\]

The compiler can consequently compute the induced norm of the factorized residual
Jacobian at the midpoint of an observed upstream perturbation, without finite
differences. This is the appropriate diagnostic for composed evaluation because it
uses the actual state pair rather than all points on the RMS sphere.

Neither certificate shows that its bound is attained on model states. Even the
tighter global coefficient (about 0.8 output-norm units per input-norm unit) is not
small enough by itself to promote the rewrap. These are structural diagnostics,
not held-out or composed behavioral evidence.

This is a genuine structural rate--distortion point but not yet a better operational
replacement. Its hash differs from the frozen product program, so it cannot inherit
held-out, composite, extraction, removal, or OOD scores. The original product-codec
price remains attached to the frozen validation roster. A later prospective runner
may evaluate the signed-square streams, but validation outcomes must not be used to
change their quantization step or roster.

Conceptually, the form connects the partially symmetric MLP tensor to signed Waring
terms, but it does not make individual square directions semantic. The distinction
between a symmetric tensor and a chosen symmetric rank decomposition follows
[Comon, Golub, Lim, and Mourrain](https://doi.org/10.1137/060661569).
