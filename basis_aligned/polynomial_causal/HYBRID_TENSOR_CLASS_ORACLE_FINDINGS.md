# Hybrid tensor-class oracle: attention is the dominant missing primitive

Date: 2026-08-28 07:44 UTC

Status: completed discovery result under the frozen protocol in
`HYBRID_TENSOR_CLASS_ORACLE_PREREGISTRATION.md`. Both compiler evaluation roles were
already spent. This result chooses a grammar; it is not strict executable recovery or
OOD evidence.

## Result

Held-out live CE is $3.09711$. At the checkpoint selected only on `skip7000`, the four
arms give:

| arm | held-out CE | harm over live |
|---|---:|---:|
| both attention and MLP compiled | 6.77256 | 3.67545 |
| attention native, MLP compiled | 5.43405 | 2.33694 |
| attention compiled, MLP native | 6.61121 | 3.51410 |
| both native | 3.09711 | 0.00000 |

Therefore restoring the exact attention contraction improves CE by

$$
G_{\mathrm{attn}}=3.67545-2.33694=1.33851,
$$

whereas restoring the exact bilinear MLP contraction improves it by

$$
G_{\mathrm{MLP}}=3.67545-3.51410=0.16135.
$$

The attention restoration is $8.30$ times larger in this compiled context. Both
restorations help, but their interaction is strongly redundant:

$$
I=3.67545-2.33694-3.51410=-2.17559\ \text{nat}.
$$

Thus the individual restoration gains are conditional effects and must not be added.
The registered attention-dominance, each-half-helps, and control predictions pass;
the registered superadditive-harm prediction fails.

## What this means

The current output compiler deletes the defining attention computation:

$$
P_{ts}=
\frac{\langle q_t,k_s\rangle\langle q'_t,k'_s\rangle}{d_h^2},
\qquad
y_t=W_O\sum_{s\le t}P_{ts}\big((1-\lambda)v_s+\lambda v^{(0)}_s\big).
$$

A token table plus a local linear map cannot express content-selected nonlocal
transport. Fixed lag features add positions but still delete the content-dependent
selection rule. The oracle says that deletion is the dominant class error.

The next compiler should therefore preserve head RMSNorm, RoPE, both QK contractions,
their product, causal mixing, the shared first-layer value bus, output projection, and
residual interface. Simplification should occur inside the six typed linear maps and
through sharing their input/output dictionaries. This is a smaller search problem than
learning an arbitrary map from residual state to attention output and is structurally
composable across sequence length.

## Controls and limits

- Both-compiled step-zero held-out recovery is $+0.38578$, exactly reproducing S1748.
- Its selected held-out recovery is $+0.58569$, within the registered S1750 window.
- Table-only CE, live CE, 5,419-token coverage, and all-site fit firing pass.
- Runtime was 231.5 seconds; row loading was not the limiting cost.
- The compiled arms still use post-forward hooks, dense token tables, and native
  fallback. They are causal probes, not compressed executables. The corrected cost
  interpretation is in `COMPILER_COST_CORRECTION_2026-08-28.md`.

Machine-readable evidence is
`../bilinear_quotient/ops/hybrid_tensor_class_oracle_results.json`.
