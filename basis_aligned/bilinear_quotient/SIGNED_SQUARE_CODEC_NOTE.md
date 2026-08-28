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

For native prefixes k=32,64,128,256,512, the resulting worst-case Lipschitz
upper bounds are 2.083, 2.771, 3.792, 4.940, and 6.445. Thus the nearly constant
relative tensor error does **not** imply a capacity-independent composition
guarantee. These bounds can certify safety if an upstream perturbation budget is
small enough, but they are deliberately conservative: they do not show that the
bound is attained on model states and are not held-out or composed behavioral
evidence.

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
