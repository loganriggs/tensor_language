# Tensor-preserving attention compiler preregistration

Date: 2026-08-28 07:44 UTC

Status: CPU implementation and protocol frozen; not queued or executed. This grants no
new row, final-role, selection, or promotion authority.

## Objective

Test whether bilin18 attention can be simplified without deleting its causal tensor
contraction. Replace the native attention module *before execution* with a standalone
program that preserves

$$
q,k,q',k'=\operatorname{RoPE}(\operatorname{RMSNorm}(Wx)),
$$

$$
P_{ts}=\mathbf 1_{s\le t}
\frac{\langle q_t,k_s\rangle\langle q'_t,k'_s\rangle}{128^2},
$$

and the exact value-bus mixture and output map

$$
y_t=W_O\sum_sP_{ts}\left((1-\lambda)W_Vx_s+lambda v_s^{(0)}\right).
$$

Only the six linear projections may be compressed. There are no token tables, output
hooks, native fallbacks, or fixed-lag substitutes.

## Phase-zero executable identity gate

Construct all six projections densely from the pinned checkpoint, discard every
reference to the native attention object, and dispatch all 18 attention calls through
`TensorPreservingSquaredAttention`. Before any compressed arm is interpreted:

1. every attention write and the first-value bus must match the native implementation;
2. final logits and covered CE must match the native baseline within numerical
   tolerance fixed by dtype replay;
3. a call ledger must report exactly 18 program calls and zero native attention calls;
4. mutation or replacement of native attention projections after construction must not
   change program outputs;
5. the storage receipt must include all six projection tensors, 18 lambdas, rotary
   constants, and code metadata, with total input support and no table values.

Failure stops the protocol. A hook that lets native attention run cannot pass.

## Discovery arms

Use the same bottom-up fit/evaluation roles as the committed routing/value curves;
they are spent and all conclusions remain discovery-only. The output projection is
dense and explicitly priced in the first sweep. Compare:

| arm | Q/K/Q2/K2 | value | output |
|---|---|---|---|
| dense identity | dense | dense | dense |
| routing-384 | rank 384 | dense | dense |
| value-384 | dense | rank 384 | dense |
| joint-384 | rank 384 | rank 384 | dense |
| joint-512 | rank 512 | rank 512 | dense |

Ranks refer to stored two-factor maps. Fit the activation-weighted projection using the
same estimator as the earlier curves; do not silently substitute weight SVD. Also
report a shared-input QK dictionary arm at rank 384, because one common encoder plus
four typed decoders costs $5Dr$ values rather than four independent $2Dr$ factors.
This arm must be trained and evaluated as a distinct registered class.

## Frozen outcomes and decisions

Let $c_0$ be native CE, $c_C$ the constant-attention CE, and $c_A$ the arm CE. Report

$$
R_A=\frac{c_C-c_A}{c_C-c_0}
$$

together with complete stored bits and measured multiply-adds. No factor-only or
conditional-table denominator is permitted.

- Identity passes only if $R\in[0.995,1.005]$ and all phase-zero gates pass.
- Projection composition passes if joint-384 harm is no more than 0.10 nat above the
  sum of the routing-384 and value-384 harms.
- Executable compression passes if an arm has $R\ge0.90$, fewer stored projection bits
  and multiply-adds than dense attention, total support, and zero native calls.
- If independent joint-384 fails but the shared-QK arm passes at matched or lower cost,
  choose shared dictionaries. If both fail, preserve more rank or learn the factors
  under final CE while retaining the same contraction; do not return to fixed lags.

Only after an attention arm passes these gates may it be crossed with the compiled MLP
program. That cross must use a factorial interaction receipt because the hybrid oracle
measured $-2.17559$ nat redundancy.

## Implementation artifact

`tensor_preserving_attention.py` is the CPU-owned executable kernel. It stores dense or
factored projections, reproduces RoPE/RMSNorm/squared QK/causal value mixing directly,
returns the first-value bus, reports complete tensor storage, rejects malformed support,
and estimates multiply-adds. Its tests prove dense numerical identity against an
independent formula and prove continued execution after native projections are made
uncallable.
