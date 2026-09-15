# Setting2 MLP16 × head17.2 cross-term fold V1

## Question

At the 128 already cached native relation endpoints, how much of each selected
MLP17/unembedding numerator is the interaction between the scaled MLP16 write and
head17.2's write? This is a no-fit algebra and magnitude audit for
`PATH-SET2-001`, not a causal circuit claim.

Let $p\in\mathbb{R}^{1152}$ be the scaled MLP16 output, let
$h\in\mathbb{R}^{128}$ be head17.2's pre-projection value, and let
$a=O_{17,2}h\in\mathbb{R}^{1152}$. For the twelve UK/US token readers
$U\in\mathbb{R}^{12\times1152}$ and MLP17 factors
$L,R\in\mathbb{R}^{4608\times1152}$ and
$D\in\mathbb{R}^{1152\times4608}$, freeze

$$
C=UD
$$

and compute the two ordered terms

$$
t_{p,a}=C\left((Lp)\odot(Ra)\right),\qquad
t_{a,p}=C\left((La)\odot(Rp)\right).
$$

Their sum must equal the direct four-corner interaction of the native MLP17
numerator, projected through the same readers:

$$
t_{p,a}+t_{a,p}
=U\left[q(p+a)-q(p)-q(a)+q(0)\right],
\qquad q(x)=D((Lx)\odot(Rx)).
$$

## Frozen predictions and decisions

1. **Exact instrument:** relative FP64 replay error is at most $10^{-10}$.
2. **Live selected interaction:** the cross-term RMS is at least 2% of the RMS of
   the full MLP17 numerator under the twelve readers on these endpoints.
3. **Both orderings matter:** the smaller ordered-term RMS is at least 5% of the
   larger. Failure permits a one-ordering path candidate; it does not invalidate
   the exact expansion.
4. **Conditional representation price:** materializing $C$, $LO_{17,2}$, and
   $RO_{17,2}$ uses fewer scalars than the direct
   $12\times1152\times128$ mixed tensor. Report this conditional price alongside
   native-factor reuse; do not call it a full-model saving.

Also report endpoint and reader distributions, cancellation between the two
ordered terms, and exact source/cache hashes. No rank sweep, coefficient fit,
activation-selected threshold, behavior/logit outcome, fresh/OOD data, gradient,
parameter update, quantization, or GPU/model forward is allowed. Down bias, other
residual/attention terms, MLP17 input RMS, final RMS, direct residual, and softcap
remain explicit omitted background. Passing this audit licenses a separately
preregistered selective causal test; it does not identify or adopt the path.
