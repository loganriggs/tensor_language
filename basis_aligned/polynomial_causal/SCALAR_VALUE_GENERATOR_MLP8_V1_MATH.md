# Absolute value generation and a simpler interaction path through MLP8

12September2026. This folds the already identified head9.8current-value reader
backward throughMLP8. Prior work compiled the finite response J_d of MLP8 to
head8.2's write. That response is reused below; the new absolute fold lets us
compare the complexity of the full value computation with its particular
producer interaction. It is not a newly discovered MLP module.

## Exact value numerator

Let r be the1152-dimensional current-value reader. MLP8 has L,R with shape
4608×1152, D with shape1152×4608, and output bias b. For the residual z entering
MLP8 define

$$
M=\operatorname{sym}\bigl(L^T\operatorname{diag}(D^Tr)R\bigr),
\qquad \rho_8^2=\operatorname{mean}(z^2)+\epsilon.
$$

M is a1152×1152symmetric quadratic coefficient matrix. If lambda0,lambda1 are
block9's residual re-entry coefficients and x0 is the normalized initial embedding,
the head9 current-value scalar is

$$
s_9=
\frac{\lambda_0\left(r^Tz+\frac{z^TMz}{\rho_8^2}+r^Tb\right)
+\lambda_1r^Tx_0}{\rho_9}.
$$

Here rho9 is the original RMS denominator of the residual entering attention9.
The formula exposes the direct, bilinear, bias and initial-state terms. The
current experiment **supplies rho9 and z from the native model**. It therefore
resolves the value numerator and moves its explanation backward one block;
it does not independently generate the complete attention9 input or its norm.
The attention9 QK computation still has its own input dependencies.

[Control and native audit](SCALAR_VALUE_GENERATOR_MLP8_V1_RESULT.json): independent
random coefficient replay2.77e-15; cached pristine/after-head8-removal value
errors4.39e-5 and5.70e-5, below the1e-4bar. These small finite-precision native
errors are not an exact FP32-operation-order claim. The change between the two
states is reproduced to9.17e-7relative error. There are960valid token positions
per state; padding was excluded. CPU execution took0.40seconds.

## Absolute structure differs from interaction structure

Diagonalize M and order its eigenvalues by absolute size. All factors are chosen
from weights. The native states only evaluate the frozen approximations.

| Kept modes | Quadratic coefficient energy | Pristine value error | Changed value error | Error in value change |
|---|---:|---:|---:|---:|
| 0 | 0% | 44.08% | 46.57% | 36.35% |
| 1 | 36.11% | 17.75% | 18.66% | 21.80% |
| 4 | 64.00% | 13.06% | 17.01% | 1.13% |
| 16 | 69.55% | 12.76% | 16.59% | 1.05% |
| 64 | 75.86% | 9.38% | 12.20% | 0.83% |

A/B hold, but C fails: no rank<=64candidate achieves<=1%error for both absolute
values and their change. Better prediction of a change is not accurate absolute
input generation, and it cannot be promoted as such.

Nevertheless, the difference is informative. Let d be head8.2's physical write
direction and a its scalar amplitude. The unnormalized quadratic changes by

$$
(z-ad)^TM(z-ad)-z^TMz
=-2a(Md)^Tz+a^2d^TMd.
$$

The mixed interaction needs the vector Md, rather than every independent entry
of M. If M has eigenpairs (mu_i,u_i), its squared directional influence has
mode contributions

$$
\|Md\|^2=\sum_i\mu_i^2(u_i^Td)^2.
$$

The [executed weight-only directional audit](SCALAR_VALUE_GENERATOR_MLP8_V1_DIRECTIONAL.json)
finds that the same leading four modes carry **99.7647%** of this directional
mass, compared with64.00%of the whole quadratic's coefficient mass. Sixteen carry
99.9404%. Thus the already identified producer writes into a much more concentrated
part of the downstream value computation than the full value map suggests.
This is a concrete composed-path observation, not a claim that M itself has rank4.

As a duplicate-work and implementation check, projecting the existing J_d bridge
into r agrees with2Md to1.38e-15. The projected mixed reader needs1152scalars.
This reuses the prior exact response algebra; it is not a separate discovery of
that identity. The bias cancels for the unnormalized quadratic difference above,
but the full normalized value response still requires changes in rho8/rho9 and
background terms. Those are retained by the earlier bridge, not silently dropped.

## Scope and next decision

The result supports decomposing **interactions between identified components**
as a unit: the directional slice is concentrated even though absolute value
preservation remains harder. It does not yet establish autonomous extraction,
full-head replacement, new behavioral selectivity or corpus OOD. These caches
include the broader original head8component removal, not a fresh test of every
later selective edit. Next validation should use the actual selective producer
and distinguish value from QK/norm ports; do not attach the observed1.13%value-change
error to full attention or logit effects without testing them.
