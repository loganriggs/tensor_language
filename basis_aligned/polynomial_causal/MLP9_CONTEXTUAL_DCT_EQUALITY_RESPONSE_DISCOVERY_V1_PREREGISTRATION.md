# Contextual DCT coverage of the natural equality/copy MLP9 response V1

## Motivation

Rung 500 prospectively established that MLP9 reads the L5H5 equality-score
relation on copy-positive tokens, rejects payload and wrong-head controls,
survives removal of the earlier service, and is copy-task selective.  Test
whether that already established semantic MLP9 response lies in the frozen
rank-16 contextual DCT output basis.

Use only documents 500--563 from the already opened rung-500 validation
partition.  Reproduce native, late-recipient-absent, and L5H5-score-restored
forwards with the rung-500 exact instrument.  At MLP9, define native response
as `late_absent_write - native_write` and restored response as
`late_absent_write - score_donor_write`.  Project each response onto the frozen
rank-16 DCT basis without fitting.  Sixteen seeded random rank-16 output
subspaces are fixed controls.

## Predictions

- **A — authority/instrument:** rung-500 has all six named predictions true;
  replayed native logits and MLP9 writes are exact; every evaluated cell has
  support; and native removal worsens copy-token NLL.
- **B — native response coverage:** DCT projection retains at least `.50` of
  native copy-positive response norm and exceeds the maximum random rank-16
  control by `.20`.
- **C — restored response coverage:** DCT projection retains at least `.50` of
  score-restored copy-positive response norm, with projected native/restored
  response cosine at least `.75`.
- **D — semantic selectivity:** DCT coverage on copy-positive tokens exceeds
  both noncopy-equality and all-noncopy coverage by `.15` for native and restored
  responses.
- **E — task bridge:** score restoration recovers `.70--1.30` of the native
  copy-token NLL effect, and its projected-response scale is positive.

Passing only licenses a prospective fresh causal projection test.  Failure
means the generic DCT node is not the known equality/copy MLP9 response under
this fixed coordinate and should not be promoted into that circuit.

## Price

One checkpoint load; 64 already opened documents; native, late-absent, and
score-restored forwards; no new text, fitting, gradients, parameter updates, or
fresh outcomes.
