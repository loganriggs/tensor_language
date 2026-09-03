# Rung 532 terminal receipt: factor substitutions work, but branch identity is not identified

**Completed:** 2026-09-03 12:38 UTC

**Independently audited:** 2026-09-03 12:44 UTC

**Owner:** Codex

## Registered verdict

The physical intervention and positive control passed. Both single-factor identities and the registered
identification claim failed, so the preregistered strong null is true. The two-factor composition screen passed.

```text
A exact live instrument                         true
B complete donor product transfers              true   (8/8 contexts)
C donor second -> target first is identified    false  (0/8 complete gates)
D donor first -> target second is identified    false  (0/8 complete gates)
E at least one held-out factor identity          false
F the two replacements compose                  true   (8/8 contexts)
strong null                                      true
```

The run used rows `500:1000`, split into `500:750` and `750:1000`, and evaluated both the 32 discovery and
30 held-out circuit-tag sets with the donor head present and absent. It executed exactly 2,625 model forwards,
zero backward passes, and no OOD rows.

## What the failed factor claims actually mean

The result is more informative than “the factors do nothing.” Each cross-branch substitution met the basic causal
transfer bars in all eight contexts:

- donor second factor into target first: minimum circuit-effect cosine `0.9566`, maximum relative error `0.3123`;
- donor first factor into target second: minimum cosine `0.8788`, maximum error `0.4781`.

Their key-permuted controls failed, so the transferred source-to-key relation matters. But the same-branch controls
also met the basic bars in all eight contexts:

- donor first factor into target first: minimum cosine `0.9543`, maximum error `0.3181`;
- donor second factor into target second: minimum cosine `0.9232`, maximum error `0.3889`.

Predictions C and D required the proposed cross-branch mapping to beat both controls by `0.15` cosine. Each beat its
key-permuted control in `8/8` contexts but beat its same-branch control in `0/8`. Therefore rung 532 cannot identify
which donor branch supplied the useful quantity. It instead raises a new, narrower hypothesis: both donor factors
may belong to one downstream-equivalent equality family, able to fill either target factor slot after a fixed scalar
rescaling. That hypothesis was not registered before these outcomes and is not claimed as a result.

The existing controls are incomplete for that new hypothesis. Rung 532 included a matched key permutation for the
two cross-branch mappings, but not for the two same-branch mappings. Rung 533 freezes all four source-branch to
target-slot mappings and gives each its own matched key-permuted control on separate natural-text and code corpora.

## Composition result

Replacing both target factors agreed with the separately scaled donor product in every context. The relative size
of the two-factor interaction was `0.075--0.250`, and the difference between the double substitution and the product
control was `0.073--0.158`, both below the registered `0.30` ceiling. This is evidence that the substitutions combine
predictably at the product level; it does not resolve factor identity.

## Integrity

- result SHA-256: `76b7c417a9bceff2f35937f51404c5248bac19b3024fb32ec6891ae70ae4ba2b`
- sufficient-statistics bundle SHA-256: `1d7e6cec94250c19159e39b24e156b0c1923fe76e364f23a69ee91b09c5a6bf0`
- frozen runner SHA-256: `142f4a0f05d582413fb6eac1820654dc6d4491690af9742e0a2d81eac719fdb8`
- managed run log SHA-256: `77934d63cb20918d1d11b0cdef203712102c84557233904af3dfa1ae8f6ef2ff`
- terminal audit SHA-256: `f6a56b3aa38e8c5eada0dab8ade8e7060ca40c3f7fb5a8e9b77554c4fa83a959`

The audit reloaded the saved sufficient statistics, recomputed every report and registered predicate, reconciled
all 2,625 forwards, reconfirmed exact native replay and factor-product identities, and verified that no raw tokens,
logits, hidden states, per-token losses, or OOD outcomes were stored in the bundle.
