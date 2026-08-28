# Causally weighted shared-routing factorization

Date: 2026-08-28

Status: mathematical fallback derived before the rank-512 discriminator. It is not yet
an empirical result.

## Why the activation metric can miss the causal metric

The current shared-QK compiler represents four row-oriented coefficient maps as

$$
C_j \approx E D_j,
\qquad j\in\{Q,K,Q_2,K_2\},
$$

with one shared input encoder $E$. It minimizes the activation-weighted error

$$
\sum_j \left\|A^{1/2}(C_j-ED_j)\right\|_F^2,
$$

where $A=\mathbb E[x^\top x]$ is the input covariance. This is the correct generalized
Eckart--Young objective for local projection reconstruction. It does not distinguish an
output direction that the suffix amplifies from one the suffix ignores.

The rank-384 whole-program result is the expected failure mode: approximately 0.02 nat
CE harm and a context-effect norm ratio near one, but a substantially rotated causal
delta. Local activation energy is therefore not the complete task metric.

## Generalized objective

Let $G_j\succeq0$ be a downstream metric on the output of coefficient map $j$. For a
Gauss--Newton causal metric, $G_j=\mathbb E[J_j^\top J_j]$, where $J_j$ maps a local
projection-output perturbation to the chosen downstream logit-delta vector. The local
surrogate becomes

$$
\mathcal L(E,\{D_j\})
=\sum_j
\left\|A^{1/2}(C_j-ED_j)G_j^{1/2}\right\|_F^2.
$$

Assume $A$ and the supported parts of $G_j$ are positive definite after a registered
ridge. Write $U=A^{1/2}E$ and fix the gauge $U^\top U=I$. For a fixed $U$, the optimal
decoder is

$$
D_j^*=U^\top A^{1/2}C_j.
$$

The optimal shared encoder subspace is given by the leading left singular vectors of

$$
\mathcal C
=\left[
A^{1/2}C_QG_Q^{1/2}\;\middle|\;
A^{1/2}C_KG_K^{1/2}\;\middle|\;
A^{1/2}C_{Q_2}G_{Q_2}^{1/2}\;\middle|\;
A^{1/2}C_{K_2}G_{K_2}^{1/2}
\right].
$$

Thus the causally weighted problem retains a closed-form simultaneous-factorization
step once the downstream metrics are estimated. The encoder is

$$
E=A^{-1/2}U_r,
$$

and the decoders use the expression above. Choosing ordered singular vectors with a
deterministic sign convention also fixes the orthogonal gauge up to degenerate blocks.

## Estimating the downstream metric cheaply

Three nested estimators should be tested in increasing cost:

1. **Single-intervention rank-one metric:** use the frozen prefix intervention and its
   reverse-mode gradient to form $G_j\approx g_jg_j^\top$. Cheap but risks fitting one
   causal direction.
2. **Small intervention bank:** change several prefix tokens/lags and accumulate
   $G_j\approx\sum_m g_{jm}g_{jm}^\top$. This directly targets context-delta recovery.
3. **Natural-output Fisher/Gauss--Newton:** Hutchinson or low-rank sketches of the
   downstream logit Jacobian on registered natural rows. More composable but costlier.

The metrics must be estimated on fit rows only. Context recovery, CE, and basis stability
are evaluated on disjoint interventions and roles to avoid merely encoding the frozen
test poke.

## Cheapest falsifier

First run the already-frozen shared rank512 discriminator. If rank512 clears context,
ordinary rank is sufficient and this objective competes only on the price frontier. If
rank512 does not clear context, fit rank384 with a small preregistered intervention bank.
The mathematical move is falsified as useful if it fails to improve held-out context
recovery by at least 0.03 without worsening either role's all-position CE by more than
0.005 nat relative to activation-weighted rank384.
