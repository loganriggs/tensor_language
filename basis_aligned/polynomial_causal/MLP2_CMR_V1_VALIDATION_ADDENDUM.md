# MLP2 CMR v1 finite-validation operational addendum

Frozen after FIT_SELECTOR calibration and before projecting the `VALIDATION` token
role, constructing a validation authority, loading validation rows into a model, or
opening any finite candidate outcome.

This addendum operationalizes quantities left underspecified by the main prospective
preregistration. It does not change the six supports, K=512 price, absolute gates,
roles, or replication rule.

## Role-only validation prerequisite

The historical token artifact contains all four roles in one PyTorch container. A
source-closed, model-free projection must validate the historical receipt and exact
`VALIDATION` tensor hashes, clone only `VALIDATION`, delete the combined object, and
publish a create-only role-only artifact/manifest/receipt. The projection process
unavoidably deserializes all roles but may not load the model, construct a candidate,
compute logits/losses, or publish any other role. The validation process may receive
only the role-only bytes and is forbidden from reading the combined container.

## Physical arms and interpolation diagnostics

The consequence arms are:

1. `NATIVE`;
2. `ZERO`, which returns a zero MLP2 write;
3. `SUFFIX`, `LOCAL`, `RMS`, `MASS`, `DERANGED`, and `HASH_RANDOM`, each retaining
   exactly its frozen 512 native channels.

For support $K$ and omitted support $S$, the physical arm is

$$
y_K(x)=b+D_{:S}\mu_S+D_{:K}\big[(L_Kx)\odot(R_Kx)\big].
$$

It must materialize only `Left[K]`, `Right[K]`, `Down[:,K]`, and the folded bias. It
may not call native MLP2 or compute an all-4,608 product vector. The fixed-grammar
price is exactly 1,770,624 stored scalar values plus 512 support indices, with
bfloat16 checkpoint-derived coefficients and products.

The signed SUFFIX diagnostics use the exact deletion direction

$$
d(x)=y_{K_{\rm suffix}}(x)-y_{\rm native}(x)
     =-D_{:S}(a_S(x)-\mu_S)
$$

and the path $y_t(x)=y_{\rm native}(x)+t d(x)$ at

$$
t\in\{-0.25,-0.10,+0.10,+0.25,+1\}.
$$

The $t=1$ arm is the physical SUFFIX candidate. The four small signed arms are
diagnostic and may compute the complete native MLP2 write and omitted-path
displacement; they receive no storage or execution credit and have separate call
ledgers.

For post-softcap logits $\ell_t$, subtract each position's vocabulary mean and let
$\delta_t=C(\ell_t-\ell_0)$, where $C$ is this centering operation. Define central
secants

$$
g_{.10}=\frac{\delta_{+.10}-\delta_{-.10}}{0.20},\qquad
g_{.25}=\frac{\delta_{+.25}-\delta_{-.25}}{0.50}.
$$

“Agrees with the suffix tangent” means, on the complete scored validation set:

$$
\cos(g_{.10},g_{.25})\ge0.90,
\quad \cos(g_{.10},\delta_{+1})\ge0.90,
$$

and for each $t\in\{.10,.25\}$,

$$
\cos(\delta_{+t},-\delta_{-t})\ge0.90.
$$

All inner products and norms sum over scored positions and vocabulary in CPU
float64. A zero norm fails rather than receiving a vacuous cosine. These thresholds
are frozen without viewing validation logits. Cellwise cosines are reported but are
not additional gates.

## Metrics and streaming sufficient statistics

All positions in the frozen eligibility mask are scored. Report nested prefixes of
48, 96, and 192 source documents; only the full 192-document result decides the
validation gate.

For every arm and registered cell, store per-document sums/counts sufficient to
recompute:

- candidate minus native cross-entropy;
- native-to-candidate teacher KL;
- centered-logit squared error and native centered-logit energy, whose square-root
  ratio is centered-logit NRMSE;
- top-1 agreement with native and top-1 accuracy against the observed target;
- $D_2=N^{-1}\sum_{i,v}(\ell^{arm}_{iv}-\ell^{native}_{iv})^2$ with a vocabulary
  sum, not mean;
- maximum document-level CE harm.

Cells are `all_scored`, nine FIT_SELECTOR target-frequency bins, `copy_positive`,
`repeat_negative`, and `nonrepeat`. Empty cells are reported as empty and cannot be
used to pass a gate. Copy masks are recomputed from validation role-only rows by the
frozen nearest-repeat rule; they are not loaded from calibration.

Raw logits, losses per token, or validation targets are not published. GPU logits
are consumed batchwise into CPU float64 document sufficient statistics and deleted.

## Margin certificate

Use exactly the 28-point epsilon grid in the calibration bundle. For each arm and
epsilon, compute

$$
B(\epsilon)=\left[1-\Pr(m_{native}\le2\epsilon)
-D_2/\epsilon^2\right]_+.
$$

Publish the complete curve, maximizing epsilon, and maximum bound. No candidate
outcome may alter the grid. The original gate requires the best bound to be at least
0.90.

## Singleton-additivity diagnostic

This diagnostic may compute all omitted products and receives no physical price
credit. On validation, report

$$
J=\frac1N\sum_i\left\|\sum_{j\in S}
D_{:j}(a_{ij}-\mu_j)\right\|_2^2
$$

and

$$
A=\frac1N\sum_i\sum_{j\in S}
(a_{ij}-\mu_j)^2\lVert D_{:j}\rVert_2^2.
$$

The ratio $J/A$ is the joint-to-singleton local distortion. `A=0` fails the
diagnostic. A ratio outside `[0.90,1.10]` is the preregistered “material
disagreement”: singleton additivity is rejected, but constant folding is not thereby
rejected.

## Shared-document inference

For each equal-price control $c$, define the pooled relative teacher-KL improvement

$$
r_c=\frac{KL_c-KL_{SUFFIX}}{KL_c}.
$$

Use 10,000 shared document bootstrap draws from the 192 document indices with
replacement, seeded by the UTF-8 string
`mlp2-cmr-v1-validation-document-bootstrap:0` through SHA-256 into a 64-bit PyTorch
CPU generator seed. Every draw recomputes each pooled ratio from per-document KL
sums and counts, then takes $r_{min}=\min_c r_c$. The simultaneous lower confidence
bound is `torch.quantile(r_min, 0.025, interpolation="lower")`. A zero control KL
fails. Gate 1 requires this lower bound to be at least 0.05.

## Absolute gates and promotion scope

On all 192 documents, SUFFIX must retain the original gates:

- $|\Delta CE|\le0.02$;
- teacher KL $\le0.02$;
- centered-logit NRMSE $\le0.10$;
- top-1 agreement $\ge0.90$;
- best margin certificate $\ge0.90$;
- no nonempty registered cell has candidate-minus-native CE above 0.02 nat;
- every signed-direction cosine above passes;
- exact price, support, gauge/permutation, physical-call, source, and receipt replay.

Validation success authorizes source-closed `REPLICATION` implementation, not a
replication outcome or a whole-model composability claim. Validation failure is
preserved and prunes K=512 native-product compression exactly as stated in the main
preregistration.
