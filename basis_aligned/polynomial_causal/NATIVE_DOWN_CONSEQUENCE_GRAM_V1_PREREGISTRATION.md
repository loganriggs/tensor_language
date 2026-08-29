# Native-Down consequence Gram v1 preregistration

## Question

Does the saved K512 MLP3 product family have a much smaller set of **physical
products** that spans its causal effects after the native Down map and native suffix?
This is different from asking whether its activations or coefficients have low MSE.

## Fixed object

For each frozen selection document, let $c\in\mathbb R^{512}$ perturb the 512 sealed
product writes by multiplying native product $j$ by $1+c_j$, so the native execution
is $c=0$. Let $R(c)$ concatenate predeclared
target/donor log-odds and suffix-state probes. Define

$$
J=\left.\frac{\partial R}{\partial c}\right|_{c=0}.
$$

Each column of $J$ is the downstream consequence of one physical product, including
its native Down column. The registered controls are: native Down, refitted Down,
same-support permuted product/Down pairings, and matched random physical columns.

## Correct randomized computation

1. Freeze a seeded Rademacher matrix $\Omega\in\{-1,+1\}^{512\times16}$.
2. Compute $Y=J\Omega$ with 16 Jacobian-vector products (JVPs).
3. Compute an orthonormal basis $Q$ of $Y$, dropping directions below the frozen
   numerical tolerance.
4. Compute $B=Q^\top J$ with one vector-Jacobian product (VJP) per retained column of
   $Q$.
5. Use the singular values of $B$ for effective consequence rank. Use pivoted QR on
   the **columns of $B$** to nominate physical products at fixed budgets.

The directional sketch $Y$ alone is not $J$ and cannot identify physical products.
No receipt may call $Y^\top Y$ the 512-product Gram. The approximated physical Gram is
$B^\top B$, and its range error must be reported separately.

## Finite-amplitude validation

For every frozen direction $v$, compute

$$
J_{\epsilon}(v)=\frac{R(+\epsilon v)-R(-\epsilon v)}{2\epsilon},
$$

$$
Q_{\epsilon}(v)=
\frac{R(+\epsilon v)+R(-\epsilon v)-2R(0)}{\epsilon^2}
$$

at both $\epsilon$ and $\epsilon/2$. Compare $Jv$ from automatic differentiation to
both central secants. This distinguishes an infinitesimal low-rank tangent from a
replacement that remains accurate at intervention scale.

## Prospective gates

All thresholds must be numerically frozen before model outcomes are opened. The run
fails if any of the following occurs:

1. JVP and the smaller-amplitude central secant disagree beyond tolerance.
2. The two amplitudes do not show stable odd response, or the even response is too
   large at the intended removal amplitude.
3. The held-out-direction error or worst registered consequence coordinate does not
   beat the refitted, permuted, and random controls at the same physical-product
   budget.
4. The selected subset does not replicate on untouched final-natural and code roles.
5. Reported executable price omits the selected product gates, Down columns, router,
   basis/indices, or precision.

The first pilot may estimate whether the method is worth a full registered run, but it
earns no strict storage, named-CE, or selective-removal credit.
