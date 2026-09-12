# When shared features and a norm suffice for a bilinear reader

12 September 2026. A weights-only extraction test following the failed
[recursive native-update pruning](SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md#recursive-extraction-fails-conditional-support-is-not-a-closed-program).

The next question is whether features can be propagated without reconstructing
the hidden state that produces them. A decomposition can have reproducible
factors and accurate conditional effects while failing this requirement. This
note supplies a local test, including normalization, and executes positive and
negative controls plus a native-weight counterexample. It is not a new fit or
an identified circuit.

## Exact local criterion

Let the columns of $B\in\mathbb R^{d\times r}$ be orthonormal and let
$Q=I-BB^\top$. Suppose the proposed state exposes only

$$
z=B^\top x,\qquad t=\|x\|^2.
$$

A normalized bilinear output reader is

$$
f(x)=\frac{x^\top Sx}{t/d+\epsilon}+b,
\qquad S=S^\top.
$$

On **all real inputs**, this output is determined by $(z,t)$ if and only if

$$
QSB=0,\qquad QSQ=\beta Q
$$

for some scalar $\beta$, with the second condition vacuous when $r=d$. For a
one-dimensional complement every symmetric restriction is already scalar;
the cross condition still matters. With these conditions, execution is

$$
f(x)=\frac{z^\top(B^\top SB)z+\beta(t-\|z\|^2)}{t/d+\epsilon}+b.
$$

To see necessity, write $x=Bz+w$, with $B^\top w=0$. The inputs $Bz+w$ and
$Bz-w$ have identical observed features and norm. Their output difference
forces the cross term $z^\top B^\top Sw$ to vanish. At fixed $z=0$ and fixed
$\|w\|$, all hidden directions must then have the same quadratic value; the
restriction of $S$ to that complement must be isotropic. Sufficiency follows
by substitution. Apply the criterion to every requested output reader.

For a fixed basis, the Frobenius-optimal isotropic coefficient is

$$
\beta=\frac{\operatorname{tr}(S)-\operatorname{tr}(B^\top SB)}{d-r}.
$$

An approximately small residual of these conditions is a probe, not automatically
a behavioral or intervention-error guarantee. The criterion is deliberately
stronger than sufficiency restricted to token-reachable states.

## A matrix-free counterexample rather than another rank sweep

For native bilinear readers,

$$
S=\tfrac12\left[L^\top\operatorname{diag}(c)R+
R^\top\operatorname{diag}(c)L\right].
$$

Compute $SB$ directly from the factors, using $O(mdr)$ arithmetic and an
$O(dr)$ cross-block output, without constructing a $d\times d$ matrix.
If $QSB$ has singular triplet $(w,\sigma,v)$, let $u=Bv$ and

$$
x_\pm=\sqrt{d/2}(u\pm w).
$$

These inputs have the same $z$, norm squared $d$, and normalized-reader gap

$$
f(x_+)-f(x_-)=\frac{2d\sigma}{1+\epsilon}.
$$

This is a concrete witness that the proposed observables omit information the
reader uses. It identifies a missing cross interaction, not its linguistic
meaning or a smallest alternative feature set.

The [native test](REGIONAL_FEATURE_FIBER_V1_RESULT.json) uses the existing
regional source block's two parent and two child **current-stream** readers
as $B$, and the already folded two regional MLP16 output readers. Their first
value stream and downstream17 normalization remain outside this local test.
This reuses the [existing MLP16 fold](REGIONAL_PAYLOAD_MLP16_FOLD_V1_RESULT.json)
and [dossier](explanations/MLP16_CURRENT_UNDERSTANDING.md); it does not redo its
earlier contribution screen or imply MLP16 dominates the regional behavior.

Both readers fail exact four-feature sufficiency. Leading cross singular
values are 9.776 and 10.927; divided by $\|SB\|_F$, they are 0.574 and 0.526.
Feature and norm equality errors are at most $1.25\times10^{-15}$; predicted
and directly evaluated output gaps agree within $1.14\times10^{-15}$ relative.
The output gaps are internal scaled reader units, **not logit nats**.

These witnesses are synthetic residual inputs and need not be token-reachable.
They reject this particular exact all-real-input interface. They do not reject
approximate closure on actual text, different linear features, nonlinear shared
features, or a simpler composed path. The native RMS-scale norm alone does not
make a synthetic input in distribution.

## The shared norm must also have an update rule

[Executed controls](NORM_AWARE_FEATURE_FIBERS_V1_CONTROL.json) include a positive
isotropic-complement example (replay error $1.57\times10^{-16}$) and a rejected
anisotropic-complement example. They also show why passing output-reader closure
is insufficient for a multi-layer executable state.

Take two inputs $x=(1,1)$ and $x'=(1,-1)$, observe their first coordinate and norm,
and update both coordinates by $x_1^2/\rho(x)^2$. The next first coordinate is
identical for the pair, but the next full squared norms differ by approximately
4. Thus a visible reader can close while the next normalization does not.
Supplying the original model's next norm would reintroduce the hidden dependency.

For a general residual bilinear update, the next squared norm contains mixed
linear/bilinear and bilinear/bilinear terms—cubic and quartic numerator terms
before accounting for bias and normalization. Those shared quantities require
their own representation and closure checks. A small output basis alone cannot
certify them.

## What the test suggests constructing

When closure fails, the exact missing terms are explicit:

$$
x^\top Sx=z^\top B^\top SBz
+2z^\top B^\top SQx+x^\top QSQx.
$$

The cross readers and complement quadratics can become candidate shared
intermediates, jointly factorized across consumers. Simply adding their native
outputs as extra features is not extraction: their computation and the norm
update must also simplify. The test gives an intervention-based criterion for
rejecting insufficient feature interfaces before investing in full-model fits.

## Relation to existing work

The project already considered closure in its
[10 September review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1349.md).
[CLUE](https://arxiv.org/abs/2004.11961) computes minimal constrained linear
lumpings for polynomial differential equations. Its preservation-of-observables
view motivates the closure question, but its algorithm and guarantees do not
directly solve this discrete normalized transformer with nonlinear norm state.
The quadratic-plus-norm criterion and witness above are derived here for this
specific object; they are not a claim that CLUE proved a transformer reduction.
