# Conditional simplicity for interventional tensor programs

This is the mathematical contract for the polynomial-causal experiments. It is a
falsifiable proposal, not a claim that neural networks have a unique intrinsic
description length.

## 1. The conditioning must be visible

Fix a specification

```
Sigma = (grammar, decoder, frozen substrate, tokenizer, task distribution,
         intervention vocabulary, numeric semantics, compiler/search budget).
```

Changing any member of `Sigma` can change what is simple. In particular, a decoder
containing the original weights or an unpriced lookup library can make an arbitrary
behavior cheap. Results therefore report `Sigma`, and rankings must be stable across
at least two genuinely different reasonable grammars before they are treated as
structural evidence.

For an original system `F`, candidate program `P`, and a priced intervention
transport `tau`, define the conditional structure function

```
C_Sigma(F; epsilon) = inf [L_grammar(P) + L(tau)]
                      subject to D_j(F, P, tau; Q) <= epsilon_j for every j.
```

`tau` maps a declared intervention on `F` to one on `P`. It is unnecessary for
interventions at a shared typed module boundary, where the identity map is fixed.
It is mandatory for latent-variable interventions: coordinates in two different
programs are otherwise incomparable and basis-dependent. A useful causal distortion
compares responses, `(P^tau(I) - P)` with `(F^I - F)`, rather than only comparing the
two intervened outputs.

The constraint vector contains, separately:

1. natural-distribution output KL/CE and task loss;
2. held-out interventional response error, using worst-case or high-quantile error
   within each intervention family rather than one pooled expectation;
3. OOD row, corpus, and token-class error;
4. composition error for simultaneous replacements/interventions;
5. collateral damage outside the targeted behavior.

Intervention families used to price or fit `P` cannot also be counted as evidence
that the price predicts intervention behavior. The present ledger fits singleton and
pair cuts and tests an unseen triple cut on disjoint rows; later tests must hold out
whole intervention families.

## 2. Simplicity is a vector

The primary report is

```
K(P) = (standalone bits, amortized/library bits, multiplication and FLOP counts,
        precision-aware interface capacity, graph locality, conditioning/robustness,
        certificate or compiler upper-bound status).
```

These coordinates answer different questions. Short description length does not
imply modular removal. A low-dimensional real interface can transmit arbitrary
information if precision is free. A low multiplication count does not imply a small
causal interface. We may scalarize `K` only after declaring an application-specific
cost vector.

The implemented JSON/zlib codec is currently an executable prototype of one upper
bound, not `C_Sigma` itself. Broad arithmetic-circuit minimization and tensor-rank
optimization are intractable, and bounded-rank tensor approximants need not exist.
Heuristic search results must be labeled with compiler version, rewrite set, and
budget. Exact language is reserved for fragments with certificates.

## 3. A certified fragment: real scalar quadratics

Let

```
q(x) = x^T S x
```

and let a multiplication gate be the product of two arbitrary real linear forms.
Only the symmetric part of `S` matters. If its inertia is `(p, q)`, the exact minimum
number of product gates is

```
max(p, q),
```

not `rank(S) = p + q`.

For the lower bound, one gate has symmetric coefficient matrix
`(a b^T + b a^T)/2`, whose positive and negative indices are each at most one. A sum
of `k` gates therefore has positive and negative indices at most `k`. For the matching
construction, pair a positive and a negative eigendirection:

```
lambda (u.x)^2 - mu (v.x)^2
  = (sqrt(lambda) u.x + sqrt(mu) v.x)
    (sqrt(lambda) u.x - sqrt(mu) v.x),
```

then use squares for unpaired same-sign modes.

The certified question slice at `mlp11` has one positive and one negative mode, so it
is one multiplication gate in this grammar while retaining a two-dimensional input
interface. The two facts are complementary, not contradictory. Vector-valued
quadratics require a harder joint partially symmetric tensor factorization; scalar
inertia must not be generalized to that setting without a proof.

### Vector-valued quadratics: the shared-product object

Between normalization boundaries a bilinear MLP is a quadratic map

```
F(x) = sum_i c_i (a_i.x)(b_i.x),     T_F in Y tensor Sym^2(X*)
```

The multiplication-minimal number of terms is the real product rank of this
partially symmetric tensor under the stated grammar. This is the right quantity
for shared vector output slices: one product is computed once and its vector
coefficient `c_i` can feed every output coordinate. It reduces to `max(p,q)` for a
scalar quadratic, but for multiple outputs it is neither the rank of any one output
matrix nor the sum of scalar minima. Simultaneous and partially symmetric ranks are
a developed algebraic object; see [Gesmundo, Oneto, and Ventura
(2019)](https://arxiv.org/abs/1810.07679).

Three cheap certificate lower bounds follow for any proposed `k`-product program:

1. `k >= rank(T_[Y | Sym2 X])`, the output flattening rank;
2. `k >= ceil(rank(T_[X | Y tensor X]) / 2)`, because one symmetrized product has
   input flattening rank at most two;
3. `k >= max(max(p_lambda,q_lambda))` over any tested output contractions `lambda`,
   using the scalar inertia theorem above.

The original hidden-unit factorization is an explicit upper bound. Equality with
any lower bound certifies a minimum. Otherwise we report an interval, not an exact
complexity. `vector_quadratic_complexity.py` implements these certificates and
gauge-regression tests on small tensors.

This reframes the next compiler experiment. For an output basis or content API,
factor the joint tensor once, price shared projections and products once, and score
the result inside the current composite. Fitting each scalar direction separately
would discard precisely the tensor-network advantage we are trying to exploit.

The general minimization remains computationally hard, and even best bounded-rank
tensor approximations need not exist. Approximate compiler outputs are therefore
upper bounds whose conditioning and achieved distortion must be recorded.

The first weights-only audit sharpens the choice of scope. On MLPs 0, 1, 2, 11,
and 17, two independent Gaussian evaluation sketches give output-flattening rank
1152/1152 at relative thresholds through `1e-4`. The smallest observed singular
value is only `0.00089`--`0.00418` of the largest, however, so full rank at every
registered threshold is mechanically implied by the measured spectra. The audit is
therefore evidence against *near-exact coefficient degeneracy at the tested scale*,
not a symbolic exact-rank certificate and not evidence of incompressibility on the
natural activation distribution. The practical coefficient-space knee was not
measured. The native 4608-product factorization remains an upper bound, while 1152
is a randomized numerical lower bound conditional on the stated tolerance. The
one-product question result survives because it is a selected scalar causal
interface. Therefore interface discovery, approximation currency, and tensor
compilation must be specified jointly.

### Exact, distributional, and causal product rank are different objects

Let `G_k` be the programs with at most `k` shared product gates. The coefficient
tensor gives the exact algebraic quantity

```
r_exact(F) = min { k : F = G for some G in G_k }.
```

For activations `x ~ D` and a declared output metric `M`, the practically relevant
approximation quantity is instead

```
r_D(F; epsilon) = min { k : E_D ||F(x)-G(x)||_M^2 <= epsilon^2 }.
```

On scalar quadratics its geometry is set by the fourth-moment operator

```
<A,B>_D = E_D[(x^T A x)(x^T B x)],
```

not by the ordinary Frobenius norm of coefficient matrices. This may be only a
seminorm: two different tensors can agree on a low-dimensional activation support.
Consequently, Gaussian coefficient sketches estimate neither `r_D` on natural
activations nor causal sufficiency.

Finally define `r_I(F; epsilon)` by replacing the natural error with the registered
worst-family error of intervention *responses*. It depends on the intervention
vocabulary and transport in `Sigma`. In general there is no ordering between a
candidate's natural and intervention error: rare causal directions can have tiny
natural mass, and high-reconstruction directions can be downstream-inert. The
compiler must therefore report the triple

```
(r_exact certificate or interval, natural-activation frontier, causal frontier),
```

and never promote one coordinate into another.

This also fixes the matched baseline for the content compiler. For original product
features `phi_i(x)=(a_i.x)(b_i.x)`, selecting `k` native units is followed by the
optimal frozen-discovery decoder regression; retaining their original output weights
would confound factor choice with decoder quality. Learned paired factors receive the
same decoder, product count, precision, and output metric. Both are then installed
and scored under held-out interventions and the whole-ship loss.

The first matched-cost causal test demonstrates why all three ranks are required.
On the selected `mlp11` question eigenpair, the exact paired gate reconstructs to
`6.3e-7` relative error and remains below `0.56%` error in bf16. The best one-square
gate has `35.4%` held-out scalar error, yet its held-out question KL is only
`6.87e-5`, or `0.39%` of the KL from deleting the rank-2 slice. It therefore failed
the preregistered causal-separation gate. Exact product geometry is a valid algebraic
certificate here, but not a necessary behavioral explanation. This scalar route is
demoted; the next compiler must earn value at a joint content interface and in the
current whole-model ship.

## 4. Polynomial boundaries

The bilinear layers and residual additions are polynomial. RMSNorm is not: its scale
contains

```
(mean(x^2) + epsilon)^(-1/2).
```

Thus exact polynomial normal forms apply only between normalization nodes, or in the
clean-frozen-gauge intervention arm. A whole-model grammar must instead do one of:

1. admit `rsqrt`/division as explicit priced primitives;
2. approximate them on a declared norm interval and certify the error;
3. freeze the gauges and restrict the claim to that intervention semantics.

Softcaps and any other analytic operations require the same treatment. Quantization
also belongs to the semantics: coefficient rescaling and algebraic rewrites can
change coordinatewise rounding. Programs with cancellation or border-rank behavior
must pay a precision/condition cost or pass a coefficient-perturbation robustness
test.

## 5. Falsification gates

At matched replacement site and natural KL, candidate prices are compared with plain
parameter bits, compressed bytes, multiplication/FLOP count, numerical rank, edge
count, and interface dimension. A proposed simplicity coordinate fails as an
explanatory metric if any of the following holds:

- it adds no held-out predictive value for unseen intervention families, OOD
  generalization, composition error, or selective-removal collateral;
- behavior-equivalent gauges and registered rewrites change it by more than 1%;
- rankings across two frozen reasonable grammars have Kendall tau below 0.8;
- apparent prediction uses the same intervention outcomes that defined the price;
- a claimed exact minimum lacks a certificate.

The synthetic certificate suite begins with linear rank, scalar quadratics stratified
by inertia (including the rank-two/one-product case), and tree tensor networks. It
then randomizes gauges, permutations, residual reassociation, common-subexpression
elimination, distributivity, and duplicate/canceling terms. This separates codec
invariance failures from discoveries about the model.

## Literature anchors

- Li and Vitanyi, *An Introduction to Kolmogorov Complexity and Its Applications*:
  description length is conditional on a reference machine; invariance is only up
  to an additive compiler constant.
- Blier and Ollivier, “The Description Length of Deep Learning Models” (NeurIPS
  2018), and Lotfi et al., “PAC-Bayes Compression Bounds So Tight That They Can
  Explain Generalization” (NeurIPS 2022): executable/compressed descriptions and
  generalization, with the decoder and coding protocol fixed.
- Hillar and Lim, “Most Tensor Problems Are NP-Hard” (JACM 2013), and de Silva and
  Lim, “Tensor Rank and the Ill-Posedness of the Best Low-Rank Approximation Problem”
  (SIAM J. Matrix Analysis and Applications 2008): computational and topological
  limits of tensor-rank minimization.
- Geiger et al., “Causal Abstraction: A Theoretical Foundation for Mechanistic
  Interpretability” (JMLR 2025): interventions between systems require an explicit
  abstraction/alignment, motivating the priced transport above.
