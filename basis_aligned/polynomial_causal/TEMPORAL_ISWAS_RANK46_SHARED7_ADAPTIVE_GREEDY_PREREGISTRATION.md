# Temporal/is-was shared-seven adaptive greedy deletion — preregistration

Start from the ordered-atlas frontier
`MLP0, MLP1, MLP2, MLP3, MLP4, MLP6, MLP7`; MLP8 is already frozen as deleted.
At each stage, jointly patch every one-module deletion on the sealed task rows and controls.

Functional feasibility requires, on both tasks, rank-four response signed projection at least `.8`,
response RSE at most `.2`, and behavior signed projection at least `.75`. Selectivity additionally
requires median control KL at most `.02` and zero top-one flips.

Before the current support is selective, an eligible deletion must be functional and have median
control KL no larger than the current support (tolerance `1e-8`). Choose lowest KL, then lowest flip
fraction, then highest minimum behavior, then highest minimum response, then canonical removed-module
order. Once the current support is selective, only a deletion that remains functional and selective is
eligible, with the same tie-breaks. Stop at the first local boundary. No thresholds or rankings may be
changed after execution begins.

## Frozen predictions

- A: authority hashes, split disjointness, finiteness, initial-support replay, and the forward-price cap pass.
- B: the greedy path reaches at most five sources.
- C: some selected stage reaches the fixed selectivity bars.
- D: the terminal support retains MLP6 or MLP7, because MLP0–4 alone missed temporal behavior.
- E: terminal median control KL improves by at least `.02` absolute relative to the original eight-bank
  value `.0527897`.

The selected result remains a discovery-set program. A separate reverse-direction OOD execution is
required before release as a bidirectional circuit. Maximum 72 forwards, zero gradients or updates.
