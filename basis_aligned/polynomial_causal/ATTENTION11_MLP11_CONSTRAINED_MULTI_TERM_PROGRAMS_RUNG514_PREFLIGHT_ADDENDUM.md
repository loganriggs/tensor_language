# Rung514 preflight addendum: fixed Q+Q2+value allocation

**Frozen:** 2026-09-03 00:32 UTC, after the registered zero-forward mismatch-covariance companion landed and before
any rung514 model outcome.

The rung514 preregistration remains unchanged except for adding one architecture-defined attention object per MLP10
branch subset:

`phi_QQV = phi_Q + phi_Q2 + phi_V`,

where every `phi` is the exact Shapley factor allocation already defined in the preregistration. This is one fixed
sum of existing registered objects; it introduces no fitted coefficient.

The reason is prospective and hash-pinned. The independent CPU companion applied its registered analysis to rung513's
already-published mismatch shares and found that all18 attention mismatch fingerprints have the same top-three
factors `{Q,Q2,V}`, with mean pairwise cosine`.993` versus term-permutation q95`.095`. This is descriptive evidence,
not a physical circuit claim. Adding the corresponding exact response object to rung514 before execution is the
direct falsification: does that fixed allocation preserve the source relation on independent splits and survive the
same held-out and physical gates?

This adds six fixed groups, changing the fixed-bank count from42 to48 and the complete group count from113,562 to
113,568. It does not change the113,520 sparse programs, candidate cap32, thresholds, split boundaries, permutation
seeds, planted tests, confirmation, intervention rules, model-forward price, or routes. The `Q+Q2+V` object is
outcome-conditioned on rung513 and must be labeled as such. It cannot pass on attribution cosine; it must pass the
same response and causal criteria as every other group.

Frozen companion evidence:

- preregistration SHA256: `164bc70dbed5098829e3efa47d9de24323d11c35a5e892014265eb2bc70f714b`;
- source SHA256: `6cc128ffb0721ffe9665a1429df6ae8e533d7a8bef995c4b7dd050fa23f9f7eb`;
- result SHA256: `150b1448cb6df0218e6db9488921de15e29c35f98ea3fb647f5cde2833bb4d04`.

If any of these files or hashes is unavailable when rung514 validates inputs, A is false and no model outcome opens.
