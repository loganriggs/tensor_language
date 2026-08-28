# Tensor-preserving attention: exact zero-native-call identity

Date: 2026-08-28 08:09 UTC

Status: completed role-free execution identity. This proves the attention replacement
boundary, not compression fidelity, strict ship recovery, or OOD behavior.

## Result

All 18 native attention modules were copied into an owned dense tensor-program bank.
During the program forward, every native attention object was physically replaced by
an uncallable sentinel. The source-closed residual dispatcher then executed exactly one
program at each site in order.

- native reference attention calls: 18/18 exactly once;
- tensor-program attention calls: 18/18 exactly once;
- literal native attention calls in the program arm: 0/18;
- MLP calls: identical native policy, 18/18 exactly once in both arms;
- program/native tensor storage: disjoint;
- first-value bus: minted once at attention0 and returned as the identical tensor object
  through attention17;
- guard restoration and inertness: pass.

Two separate numerical gates pass bitwise in fp32:

1. Each dense program receives the exact native per-site normalized state and incoming
   value bus offline. Every attention write and returned bus matches exactly.
2. The full all-program trajectory and full native trajectory produce exactly equal
   attention writes, buses, logits, and synthetic-token CE.

The final logit hashes are identical,

```text
ecd96381a8d062a09b7d6387224fb6bf1c9dde9924952d4dec05ffbca50d09c9
```

with maximum absolute logit error 0.0. Output shape is `[4,256,50304]`; synthetic CE is
12.686808586120605 in both arms.

## Cost meaning

The dense attention bank stores 143,328,402 values, including all six dense projection
maps at each layer, lambdas, and rotary constants. It has total input support, zero token
tables, and zero native attention calls. This is approximately the native attention
projection storage, so the identity point is not itself simpler. It establishes a valid
executable denominator and a typed boundary on which routing/value compression can now
be measured.

Embedding, unembedding, residual mixing, and all MLPs remain native and are outside this
component receipt. A future whole-model cost must include them or replace them explicitly.

## Limits and next gate

The fixture is deterministic synthetic `[4,256]` input and opens no corpus role. Exact
operator identity makes this sufficient for the implementation boundary, but it says
nothing about compressed projection fidelity. The frozen next experiment fits the same
activation-weighted routing/value rank classes used by the earlier curves, executes
them through this bank with no native attention calls, and measures their joint CE and
complete storage/FLOPs.

Machine-readable evidence, including checkpoint/source/tensor hashes, is
`tensor_preserving_attention_identity_results.json`.
