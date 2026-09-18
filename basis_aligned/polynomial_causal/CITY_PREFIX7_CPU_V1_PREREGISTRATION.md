# Direct upstream CPU prefix capture

Run native blocks0–7 on forty opened Pile sequences, one padded batch, two CPU
threads,180second cap. Padding is only to the right; causal attention leaves
valid-position states independent of padding. CUDA_VISIBLE_DEVICES must be empty,
all model parameters CPU, CUDA uninitialized. No GPU service/queue mutation.
This is a new prefix-state capture, not a rerun/replacement of pending native GPU
reader and full-suffix certificates. No blocks8–17 or output logits are evaluated.

Capture residual6, initial state, mixed7, normalized attention7 input, attention7
output, normalized MLP7 input, MLP7 output, residual7 and inherited first values.
Save only original-length positions. This permits an exact attention7 generator
from its earlier state, rather than relying on subtractive recovery of residual6.

Predictions on every sequence:
- a: CPU residual7 against saved GPU residual7 relative L2<=1e-4.
- b: reconstructed four mixed8 city sources against saved GPU source7 capture
  relative L2<=1e-4 for each source.
- c: inherited first values replay the weight-derived token table<=1e-4 relative;
  all capture tensors finite; forty fixtures; eight CPU block calls only.
- d: directly captured normalized MLP7 city input agrees with the earlier
  source-reconstructed input<=1e-4 relative.

Preserve individual differences and failed gates. This tests CPU/GPU and batch
precision only at the upstream boundary; it does not certify full-model behavior,
selective edits, fresh transfer or composition. Do not silently use CPU states as
bit-exact substitutes for GPU captures.
