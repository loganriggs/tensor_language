# MLP9 equality consumer-response basis discovery V1

## Motivation

The frozen generic contextual-DCT rank-16 basis captures only `.176/.182` of
the established native/restored equality-copy MLP9 responses and is not
copy-selective.  The equality behavior and exact instrument remain live.  Test
the briefing's consumer-conditioned alternative: discover output coordinates
from the known semantic response rather than from behavior-blind curvature.

Use opened rung-500 documents 500--563 for discovery and 564--627 for
confirmation.  Fit an output covariance only from the native copy-positive
MLP9 response `late_absent - native` on discovery.  No NLL, restored response,
noncopy response, or confirmation vector may affect the eigenspace.  Report
fixed ranks `4,8,16,32,64`.  Select the smallest rank whose discovery native
copy coverage is at least `.80`; if none qualifies, select nothing.

## Predictions

- **A — instrument:** exact native replay, positive native copy-token NLL
  effect, and nonempty support hold in both panels.
- **B — low-rank transfer:** a selected rank exists, is at most 32, and retains
  at least `.70` confirmation native-copy response norm.
- **C — reuse across causal backgrounds:** the same basis retains at least
  `.65` confirmation score-restored response norm, and projected native versus
  restored response cosine is at least `.75`.
- **D — semantic selectivity:** on confirmation, native and restored copy
  coverage each exceed both noncopy-equality and all-noncopy coverage by `.10`.
- **E — generic-basis improvement:** at rank 16, confirmation native and
  restored coverage each exceed the frozen contextual-DCT rank-16 basis by at
  least `.30`.
- **F — task stability:** score restoration recovers `.70--1.30` of the native
  copy-token NLL effect in both panels.

Passing yields a behavior-conditioned response coordinate for later fresh
causal projection.  It does not yet establish extraction of its upstream
source, fresh OOD prediction, removal, or a complete equality circuit.

## Price

One checkpoint load; 128 already opened documents; four exact forwards per
four-document batch; one 1152-dimensional covariance eigendecomposition; no
new text, gradients, parameter updates, or fresh outcomes.
