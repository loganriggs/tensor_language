# V23 direct residual-readout factorial v2 — float64 telescope repair

V1 preserved all four paths and scientific arms but was invalid because its state-closure diagnostic
formed a cancellation-prone final-state telescope in float32. The max residual difference was exactly
`0.0078125`, while decoding the reconstructed state replayed rescued logits within `5.72e-6`. V2
makes one instrument change: define direct, response, and reconstructed final states in float64 for
the algebraic closure, then cast each already-defined decoder arm to float32. It reruns all four model
forwards. No population, path, carry coefficient, output arm, metric, threshold, prediction, or price
changes.

In particular, v1's descriptive direct arm leaked `0.155885` on P against the frozen `0.15` bar.
V2 does not relax or round that bar; prediction B is expected to remain false if the value reproduces.
The original v1 receipt remains `invalid_instrument` and is not relabelled.

