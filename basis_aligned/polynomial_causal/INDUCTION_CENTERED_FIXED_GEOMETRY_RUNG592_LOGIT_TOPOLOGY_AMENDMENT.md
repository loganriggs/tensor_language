# Rung 592 prospective logit-topology and evidence-price amendment

**Frozen:** 2026-09-04 UTC, before any repaired R592 implementation, model call, or outcome

**Status:** prospective narrow specification correction; implementation remains blocked pending independent approval

## Authority and reason for this amendment

The exact R592 implementation review at commit
`1c07919a4dcb37e86474999a944abdf06cb99156` blocks candidate
`0bd259b7d5a499a863741338f8b55dc11368f344` before execution. The review and its CPU test packet have SHA-256 hashes:

- implementation review: `9b8e4ce54d1b34d650ef088f841672cf01a4482257446b611ba37e1353a457cf`;
- implementation review tests: `3f8a559a14015498d375ba75271cf57647b9cc9841ef32b1e9e32406abf71323`.

The approved R592 lineage remains:

- preregistration commit `cb81a22bf10fc46e2c851361d2a5de95dd5b7045`;
- executable-contract amendment commit `eaeee8e7cd728a345a5e24421ab6aeccef4fefae`;
- diagnostic-prefix amendment commit `3be7c21c3886502ea989efdaeba5c137aef45d8e`;
- nonfinite-mask amendment commit `6d779ae45b68ffa4c3e7bdf58963cb7f7c2ed2d2`.

Those documents registered full-vocabulary logit arrays with last dimension 50,257. That constant is incompatible with
the pinned observed-model topology. The pinned facade, SHA-256
`b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`, declares
`LOGIT_VOCAB = 50_304`, verifies checkpoint configuration vocabulary size 50,304, and returns all 50,304 logits from
`forward_with_dispatch`. A 50,257-element destination therefore cannot contain one completed checkpoint response.

This amendment supersedes only the incompatible vocabulary dimension and arithmetic that follows from it. It does not
authorize implementation work or execution until independently approved.

## Exact supersession

Every R592 occurrence that means the width of a model logit vector, full-vocabulary difference, structural-output
comparison, or vocabulary RMS is **50,304**, not 50,257. In particular:

1. Each completed endpoint, directed-native, replay, score, payload, or joint diagnostic call saves `logits.npy` as
   little-endian float32 with shape `[b,50304]`, where `b` is the actual registered batch size 32 or 16.
2. Complete phase evidence saves `logit_differences.npy` as little-endian float32 with shape
   `[N_d,4,50304]`. The difference-axis order remains exactly
   `(native_minus_replay, score_minus_replay, payload_minus_replay, joint_minus_replay)`.
3. Every elementwise native/replay and structural-output identity compares all 50,304 checkpoint logits. No slicing,
   tokenizer-vocabulary prefix, padding of a shorter vector, or exclusion of logits is permitted.
4. Every saved `vocab_size` field is the integer 50,304. Vocabulary RMS is exactly

   $$
   \sqrt{\frac{1}{50304}\sum_{j=0}^{50303}(\ell^{\rm arm}_j-\ell^{\rm replay}_j)^2},
   $$

   with the inherited float64 aggregate arithmetic. FIT-frozen vocabulary scales use the same corrected RMS.
5. The nonfinite-mask rule applies to the complete `[b,50304]` raw logit array. Its one-to-one mask therefore has that
   exact shape when logits are affected.

The old 50,257 constant likely reflected a tokenizer-facing vocabulary count, but R592 intervenes on and audits the
actual checkpoint output tensor. Treating only 50,257 coordinates as “full vocabulary” would be a different
scientific object and is not authorized here.

## Recalculated raw evidence bytes

The following are array-data bytes, excluding the fixed NumPy file header. One float32 logit row costs
`50,304 × 4 = 201,216` bytes. A 32-row diagnostic `logits.npy` therefore contains 6,438,912 data bytes; the registered
16-row SELECT tail contains 3,219,456 data bytes.

For complete raw replay-relative logit differences:

| phase | exact calculation | corrected data bytes |
|---|---:|---:|
| FIT | `3744 × 4 × 50304 × 4` | 3,013,410,816 |
| SELECT | `1872 × 4 × 50304 × 4` | 1,506,705,408 |
| maximum | sum | 4,520,116,224 |

The earlier values 3,010,595,328 FIT and 1,505,297,664 SELECT are superseded. The increase is 2,815,488 FIT bytes,
1,407,744 SELECT bytes, and 4,223,232 bytes at maximum.

The other principal raw arrays are unchanged:

- centered hook deltas: 276,037,632 FIT plus 138,018,816 SELECT = 414,056,448 bytes;
- directed live projected-content factors: 138,018,816 FIT plus 69,009,408 SELECT = 207,028,224 bytes.

Consequently the corrected maximum principal raw audit payload is

$$
4{,}520{,}116{,}224+414{,}056{,}448+207{,}028{,}224
=5{,}141{,}200{,}896\ \text{bytes}.
$$

The earlier 5,136,977,664-byte total is superseded. Smaller endpoint arrays, selected-logit JSON records, manifests,
NumPy headers, receipts, and filesystem metadata remain outside this principal-array subtotal exactly as before.

## Everything else remains frozen

This correction changes no row, direction, endpoint, semantic role, site, machine arm, centered-factor formula,
support rule, factor-transport check, operational tolerance, target/control cell, bootstrap identity or draw, scientific
threshold, FIT-first opening rule, terminal precedence, publication rule, claim boundary, or call price. The prices
remain exactly 639 FIT, 322 SELECT, 961 maximum, zero backward passes, and zero weight updates. FINAL and OOD remain
closed.

The six findings in the implementation review remain blocking. After independent approval of this exact amendment, a
repaired candidate must additionally pass the authoritative repository gate, hard-abort on missing mandatory
observations, content-bind every invalid-evidence byte, preserve independently reconstructible native-attention gate
evidence, and explicitly fsync closed memmap files before publication. None of those implementation repairs is made or
authorized by this document.

## Review gate

An independent model-free reviewer must verify the pinned facade topology, every corrected shape and byte count, the
50,304-coordinate RMS definition, and the narrow supersession boundary. Only a committed APPROVE of these exact bytes
permits the five remaining implementation repairs. R592 remains prohibited from model, Torch, CUDA, GPU, queue, and
outcome access until the repaired implementation itself receives a later different-agent exact-byte approval.
