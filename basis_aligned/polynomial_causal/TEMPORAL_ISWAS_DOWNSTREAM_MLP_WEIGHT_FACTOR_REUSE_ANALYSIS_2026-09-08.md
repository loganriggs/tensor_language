# Downstream MLP weight-factor reuse analysis — 2026-09-08 10:28 UTC

## Decision this resolves

If the queued cumulative module or residual/module factorial identifies M11, M12, M15, or another
MLP as a material carrier of the compact `L7H7 + L9H4` is-was writer effect, the next circuit step
must split that native module and translate the causal response into actual weights. This note
fixes the appropriate existing mechanism before the current outcomes are known. It does not select
an MLP, fit a subspace, or change either queued experiment.

## Prior-art search and lessons

Two existing result families are directly relevant.

1. `aspectual_tense_complete_mlp_weight_tensor_diagnostic_v1` contracted a static downstream read
   map through `Down`, `Left`, and `Right`. It improved some rank correlations but failed the
   registered is-was ordering and terminated `null`. Therefore a static weight norm is only an
   incidence prior; it cannot select a causal downstream MLP or factor for this branch.
2. `iswas_mlp8_complement_product_factor_program_v1` implemented the exact activation-conditioned
   bilinear factor identity and a physical pre-`Down` intervention. Its instrument and full replay
   passed, but the two-term left+right approximation and the claim that the bilinear interaction
   was secondary both failed. Therefore the three exact factors must remain separate prospective
   arms; the interaction may not be discarded from analogy or magnitude alone.

The audited five-MLP weight-compiler line supplies a third caution: exact weight compilation and
gauge checks passed, while hidden participation remained broadly distributed. The next split must
test causal factor transfers, not convert a full-rank exact map into a sparsity claim.

## Exact reusable algebra

For MLP layer `l` and normalized input `x`, write

```text
L = Left_l(x)             shape [batch, position, 4608]
R = Right_l(x)            shape [batch, position, 4608]
h = L * R                 coordinatewise product
y = Down_l(h) + b_l       shape [batch, position, 1152]
```

Let subscripts `0` and `1` denote native and selected-writer executions. Then

```text
delta_h = h1 - h0
        = (L1-L0)*R0 + L0*(R1-R0) + (L1-L0)*(R1-R0)
        = left_change + right_change + bilinear_interaction

delta_y = Down_l(delta_h).
```

This is an exact identity, not a linearization. Installing `h0 + delta_h` at the input of the
native `Down_l` reproduces the selected-writer MLP output because `Down_l` is linear and its bias
cancels. Installing the three terms singly and in all eight subsets gives an exact finite
factorial of what changed inside the causal MLP response. The factors are invariant under the
native per-coordinate reciprocal gauge `Left_i -> c_i Left_i`, `Right_i -> Right_i/c_i` and under
paired coordinate permutations; no arbitrary neuron importance or SVD basis is required.

For a multi-MLP prefix, apply the selected factor subsets at every retained MLP's pre-`Down` input
in causal order. Full-factor installation at all retained MLPs must replay the already measured
complete-module prefix. This is the route-agreement control that prevents a locally exact algebra
from being mistaken for an end-to-end causal program.

## Result-conditioned successor

Only after the queued results select a branch:

- If module response is material and a stable compact prefix exists, retain the selected MLPs and
  the already identified A11H3 branch. Run the exact three-factor factorial per selected MLP plus
  the full selected-prefix replay. Original FIT may choose factor groups; HOLDOUT cannot reselect.
- If module response is material only as the full P16/R0M1 set, first use leave-one-MLP and
  attention-head splits rather than a 16-module product-coordinate sweep.
- If residual identity dominates, do not run this MLP split. Translate the block-10 boundary
  state through residual coefficients and the exact final readout instead.
- If neither partial arm is material, switch to a state×readout causal-response object; a static
  weight tensor, lower rank, or more hidden-coordinate sparsity cannot resolve that interaction.

Promotion requires exact full-factor replay, held-out task-effect geometry, temporal selectivity,
and a subsequent OOD/removal test. A factor screen is not yet an identified or adopted circuit.

## Price and implementation reuse

The existing `capture_mlp8`, `run_hidden_patch`, and exact factor-closure patterns can be generalized
by layer rather than rewritten. For `k` selected MLPs, a basic screen needs native/writer capture,
one complete-prefix replay, three singleton factor arms per MLP, and selected joint factor arms;
the exact count will be frozen only after `k` is known. No backward pass, optimizer, DAS projector,
rank choice, or fitted parameter is necessary.
