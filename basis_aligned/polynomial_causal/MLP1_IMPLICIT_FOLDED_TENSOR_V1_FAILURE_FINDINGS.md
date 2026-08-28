# MLP1 implicit folded tensor v1: fail-closed launch finding

Date: 2026-08-28

Status: no scientific tensor outcome. The v1 namespace is spent and must not be
reused.

## What happened

The source/weight authority was frozen successfully and independently audited. Its
SHA-256 is
`7b2cebe982559a3e232e073d238b984df6246461b92c622e0275bc9279b8b468`.
The collector then stopped during full checkpoint state-tree validation, before
copying MLP1 factors or computing a mode Gram, spectrum, or projected core.

The exact error was:

```text
checkpoint state-tree metadata changed: transformer.wte.weight
```

The create-only failure receipt records no partial result and explicitly marks the
result unauthorized.

## Diagnosed cause

The validator constructed a meta model under PyTorch's default float32 dtype and
incorrectly required every checkpoint tensor to have the corresponding meta-model
dtype. The checkpoint has the same 218 keys and the same shapes, but intentionally
uses mixed serialization dtypes:

- MLP Left, Right, and Down weights are float32;
- token embeddings, residual/attention scalar parameters, and every MLP
  `Down_bias` are bfloat16.

Thus the first mismatch appeared at the token embedding before the validator
reached MLP1's bfloat16 bias. The v1 collector also incorrectly required that bias
to be float32. This is a source-schema error, not a tensor result and not evidence
about MLP1 compressibility.

## Retry boundary

A retry requires a new v2 namespace and prospective source closure. It must:

1. bind the v1 authority and failure receipts as immutable prior history;
2. validate all checkpoint keys and shapes against the meta model while validating
   dtypes against the exact mixed-dtype serialization contract;
3. require float32 MLP1 Left/Right/Down and bfloat16 `Down_bias`;
4. hash and record the original bfloat16 bias, then convert only an owned analysis
   copy to float64;
5. keep bias out of every folded tensor, Gram, spectrum, and core calculation;
6. fail tests for any unregistered key, shape, device, or dtype change.

No v2 authority or result may be opened until that corrected source is committed,
pushed, and independently audited.
