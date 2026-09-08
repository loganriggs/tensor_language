# P7 physical-to-weight tensor translation plan — 2026-09-08 12:39 UTC

## Why this follows the causal split

The established object is an exact OOD-composable physical interface: block-10 residual identity
plus the fixed P7 downstream response. The queued LOO audit and conditional A11/M11 experiments
refine that interface into operational pieces. Only pieces that pass exact route replay and stable
causal bars should be translated into checkpoint weights; translating every correlated activation
would recreate the readout-not-cause and DAS-overfitting failures.

The A11 and M11 conditional experiments use the same OOD bank as their module-level gate. Because
H3 and M11 were nominated before that gate, their registered tests are confirmatory for those
specific hypotheses, but not a fresh-corpus replication. Any newly discovered head/factor remains
diagnostic until a sealed construction or cross-task intervention confirms it.

## Attention write tensor

Let `z[h,a]` be A11's concatenated pre-`c_proj` head tensor with head index `h in 0..8` and private
coordinate `a in 0..127`. With output weight `W_O[r,h,a]` and residual index `r in 0..1151`, a
passing head contribution has exact written residual tensor

```text
delta y_A11[r] = sum_a W_O[r,h,a] * delta z[h,a].
```

This contraction is invariant to a paired value/output change of basis inside the head. It turns a
passing H3 intervention into actual weights and a row-conditioned written tensor. Downstream
attention/MLP reader compatibility is then obtained by contracting `delta y_A11[r]` with the
relevant normalized-input Jacobian and `W_Q/W_K/W_V` or `W_Left/W_Right`; static compatibility is a
candidate edge, and physical reader-side interchange/removal is the identifying test.

## MLP product-factor tensor

For hidden coordinate `i in 0..4607`, the conditional M11 executor defines exact arm-local factors

```text
f_left[i] = delta L[i] * R0[i]
f_right[i] = L0[i] * delta R[i]
f_int[i] = delta L[i] * delta R[i].
```

With checkpoint output tensor `Down[r,i]`, each factor writes

```text
delta y_factor[r] = sum_i Down[r,i] * f_factor[i].
```

These are literal checkpoint-weight contractions, not fitted directions. Reciprocal
`Left_i/Right_i` rescaling and paired hidden-coordinate permutation leave the three written tensors
well defined. The exact all-three equality supplies a route certificate from hidden factors through
`Down` to the already validated complete M11 response.

## Shared subspace and circuit edges

Across tasks or hypotheses, collect only causally passing residual writes as columns of
`Y in R^[1152,k]`. A shared output span or downstream response span can nominate a shared circuit
variable, but principal angles or tensor overlap are diagnostic. Identification requires:

1. a downstream reader intervention that accepts either task's matched write and predicts its
   signed effect on held-out text;
2. an upstream writer intervention whose checkpoint-weight contraction reproduces that write;
3. joint installation with task-specific branches that preserves the measured composition law;
4. selective removal or editing with low collateral.

This is how repeated task subspaces become concrete upstream-writer and downstream-reader edges.
The weight tensors reduce the search space and make the proposed computation legible; causal
cross-installation decides whether the shared span is genuinely one reusable subcomputation.

## Executable continuation

- Consume the immutable P7 module-necessity receipt.
- If A11/H3 passes, store its exact `W_O`-contracted per-row writes and rank downstream reader
  contractions, then preregister the top physical reader edge before testing it.
- If M11 passes, store all three `Down`-contracted write tensors and test which factors downstream
  readers accept; preserve a failed interaction or linear factor as an exact null.
- Compare passing A11/M11 output tensors for a shared response span only after causal factor results
  land. Do not fit a shared DAS basis on this already opened OOD bank.
