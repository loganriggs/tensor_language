# MLP1 implicit folded-tensor v2 dtype retry erratum

Date: 2026-08-28

Status: prospective retry specification only. This document authorizes no checkpoint
deserialization and no scientific outcome. V2 requires a new create-only authority
after its complete source closure is committed and pushed.

## Preserved v1 failure

V1 authority SHA256 is
`7b2cebe982559a3e232e073d238b984df6246461b92c622e0275bc9279b8b468`.
V1 failure SHA256 is
`350d7dc7bb4ec3207853abaa5db83da57b58b1613898553320824245c3f48526`.
The failure is nonauthoritative, binds that exact authority, has no partial result,
and reports `checkpoint state-tree metadata changed: transformer.wte.weight`.
The v1 result and outcome-authority paths are absent. V2 never overwrites, deletes,
or semantically relabels those facts.

## Narrow repair

The v1 meta-model correctly fixed all 218 state keys and shapes but incorrectly
assumed every serialized tensor was float32. The checkpoint-native dtype schema is:

- bf16: `transformer.wte.weight`, every block `lambdas`, every `attn.lamb`, and
  every `mlp.Down_bias`;
- float32: every other state key.

Consequently MLP1 `Left.weight`, `Right.weight`, and `Down.weight` are float32,
while MLP1 `Down_bias` is bf16. V2 changes only this metadata validator and the
bias-copy receipt. All numerical ranks, blocks, thresholds, projected-core supports,
prices, publication order, and claim boundaries are inherited exactly from v1.

The original MLP1 bf16 bias is cloned, hashed, and recorded before any conversion.
A disjoint float64 analysis copy is then made and separately hashed. Only the copy is
passed to the pure diagnostic, where bias remains separate. Changing bias may change
only the reported bias provenance/norm; it must not change balancing, Down spectra,
mode Grams, HOSVD spectra/bases, projected cores, or price curves.

This is a versioned execution correction, not an outcome-dependent scientific change.
