# Temporal/is-was rank46 shared-eight pairwise Möbius pilot — preregistration

## Fixed question and authority

The joint-source factorial identified a jointly sufficient shared bank of eight physical MLP modules:
`MLP0, MLP1, MLP2, MLP3, MLP4, MLP6, MLP7, MLP8`. Its receipt SHA-256 is
`3f4a1a1260d5b1493bc14cca74c03f1dd03523e773b6fb8b8c197da7a6c31e5f`.
This run asks whether the atlas's large singleton overcount is explained by sparse degree-two causal
interactions within that bank.

The source subsets are frozen before execution: empty, all eight singletons, all 28 unordered pairs,
and the full eight-module intersection. The response basis, task-rank-four modes, row splits, behavior
margin, and controls are inherited unchanged from the authority runner. No optimization, gradients,
threshold search, or model update is permitted.

## Frozen predictions

- A: authority identities match; the full shared-eight replay remains finite and has response projection
  at least `.8` and behavior projection at least `.75` on both tasks.
- B: singleton non-additivity reproduces prospectively: singleton-sum projection is at least `1.5` and
  singleton relative squared error at least `1.0` for both tasks.
- C: at least one pair Möbius coefficient has norm at least `.05` of the donor-response norm on each task.
- D: adding all pair coefficients reduces singleton relative squared error by at least `.10` absolute on
  each task. This is the degree-two adequacy gate.
- E: the eight pairs with largest pooled norm contain at least `.50` of total pooled pair-norm mass. This
  is the sparse-concentration gate.

A failure is retained as an honest null. No bar may be relaxed. If D and E pass, the top pairs become
atomic groups in a prospectively frozen pooled greedy deletion. If D fails, stop global pair expansion
and test ordered layer-band conditional increments. If D passes and E fails, use ungrouped pooled
backward deletion because interactions are dense.

## Price

Exactly 38 patched source subsets, plus native base/donor captures and one full-intersection control
capture. Maximum 48 model forwards, zero backward passes, zero fit updates, and zero model updates.
