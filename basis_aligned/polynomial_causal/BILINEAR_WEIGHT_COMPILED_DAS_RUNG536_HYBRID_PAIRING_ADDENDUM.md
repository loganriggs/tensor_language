# Rung 536 addendum: exact hybrid pairs for token-only and token-by-context interchange

**Frozen:** 2026-09-03 13:42 UTC, before any real-model backward pass or DAS fit

## Why use hybrid pairs?

A donor document normally changes both its current token and its context. Swapping from such a donor would not tell
us whether a learned subspace carries the token-only computation or a mixture of token and context. MLP0's exact
bilinear form lets us construct controlled hybrid inputs that vary one source at a time.

Write the normalized MLP0 input as token part $p$ plus context part $q$, and define the product activation

$$
g(p,q)=\bigl(W_L(p+q)\bigr)\odot\bigl(W_R(p+q)\bigr).
$$

It has the exact split

$$
g(p,q)=g_T(p)+g_I(p,q)+g_C(q),
$$

where

$$
\begin{aligned}
g_T(p)&=(W_Lp)\odot(W_Rp),\\
g_I(p,q)&=(W_Lp)\odot(W_Rq)+(W_Lq)\odot(W_Rp),\\
g_C(q)&=(W_Lq)\odot(W_Rq).
\end{aligned}
$$

The deployed output directions are $T=W_Dg_T$, $I=W_Dg_I$, and $C=W_Dg_C$.

## Token-only controlled pair

For a base $(p_b,q_b)$ and donor token $p_d$, keep the base context fixed and form the hybrid $(p_d,q_b)$. Its full
product difference is

$$
\Delta g_{\rm token}^{\rm hybrid}
=g(p_d,q_b)-g(p_b,q_b)
=\Delta g_T+\Delta g_I.
$$

The exact target changes only the token-only computation:

$$
g_T^{\rm target}=g(p_b,q_b)+g_T(p_d)-g_T(p_b).
$$

A learned projector $P$ is asked to satisfy

$$
g(p_b,q_b)+P\Delta g_{\rm token}^{\rm hybrid}
\approx g_T^{\rm target}
$$

under the registered downstream causal loss. Thus the search must separate the token-only change from the
simultaneous token-by-context change induced by changing the token.

## Token-by-context controlled pair

For context donor $q_d$, keep the base token fixed and form $(p_b,q_d)$. Then

$$
\Delta g_{\rm context}^{\rm hybrid}
=g(p_b,q_d)-g(p_b,q_b)
=\Delta g_I+\Delta g_C.
$$

The exact target changes only the cross term:

$$
g_I^{\rm target}=g(p_b,q_b)+g_I(p_b,q_d)-g_I(p_b,q_b).
$$

The corresponding projector condition is

$$
g(p_b,q_b)+P\Delta g_{\rm context}^{\rm hybrid}
\approx g_I^{\rm target}.
$$

## Claim boundary and controls

These targets are portable because they are exact functions of $(p,q,W_L,W_R,W_D)$, not labels inferred from the
old census. The real objective will compare downstream logits or cross-entropy after the projected intervention with
the logits or cross-entropy after the exact target intervention. Product-activation MSE is diagnostic only.

The token-only and token-by-context targets remain separate experiments. Each needs dimension-matched random
projectors, native-coordinate subsets, shuffled donor identities, held-out tokens/contexts, natural-to-code OOD,
and the frozen 32/30 circuit-response split on its original census rows. A small dimension cannot pass by itself.
There may be no fixed low-dimensional projector that performs either separation; that is the registered null.
