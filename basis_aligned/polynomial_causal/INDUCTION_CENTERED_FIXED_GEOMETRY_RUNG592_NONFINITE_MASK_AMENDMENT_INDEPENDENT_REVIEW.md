# Independent review: R592 nonfinite-mask amendment

Date: 2026-09-04 UTC

Reviewed commit: `6d779ae45b68ffa4c3e7bdf58963cb7f7c2ed2d2`

Reviewed amendment SHA-256:
`f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc`

Verdict: **APPROVED for prospective implementation**

The amendment closes the sole blocker in the diagnostic-prefix contract. It
defines an injective deterministic mask path for every affected raw array,
binds an exact array-to-mask index, and makes both missing masks and masks for
finite arrays invalid. No scientific row, intervention, gate, threshold,
bootstrap, call, complete evidence shape, or price changes.

This was an exact-byte, model-free, outcome-blind review. I did not inspect an
R592 implementation or outcome, load a model, use CUDA/GPU or a queue, or alter
the amendment, its parents, R590, or any implementation file.

## Exact authorities

The working-tree amendment is byte-identical to the blob in reviewed commit
`6d779ae45b68ffa4c3e7bdf58963cb7f7c2ed2d2`. It correctly pins:

| authority | SHA-256 |
|---|---|
| diagnostic-prefix amendment, commit `3be7c21c3886502ea989efdaeba5c137aef45d8e` | `f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62` |
| blocking diagnostic-prefix review | `e7373c2249e0456327d386559d4f3fa68e0661ed076a35fb120ad9d8effaa675` |

The parent review's sole failure was that several affected arrays all required
the same flat `nonfinite_mask.npy`. This amendment explicitly replaces only
that clause and forbids the flat name.

## Reconstructed contract

For each raw float file `{stem}.npy` in the final failing completed call, the
mask exists if and only if the raw array contains NaN or infinity. Its only
legal path is:

```text
nonfinite_masks/{stem}.mask.npy
```

The mask is a C-contiguous NumPy boolean array with the same shape and C-order
coordinates as the raw array, and equals `~isfinite(raw)` elementwise. Since a
NumPy boolean occupies one byte, its logical payload length is the product of
the dimensions.

`nonfinite_mask_index.json` is nonempty and sorted by `raw_filename`. Every
entry has exactly the nine frozen fields and reconstructs:

- the raw and deterministic mask filenames;
- raw and mask dtypes;
- common shape and boolean payload byte length;
- exact mask-file SHA-256;
- positive mask sum; and
- first true C-order coordinate.

The index raw-name set must equal the set found by loading every raw float array
in the failing call and applying `~isfinite`. Mask paths and files must also be
unique and exact. Integer, boolean, JSON, and token arrays cannot enter the
index. Absolute, parent-traversing, duplicate, missing, extra, aliased, or
noncanonical paths fail integrity.

If the predicate is anything except `nonfinite_observation`, both the index and
mask directory must be absent. For `nonfinite_observation`, both exist and the
index is nonempty. The inherited parent still limits nonfinite bytes to the
last completed call of an invalid diagnostic and forbids promotion to a normal
scientific result.

## Adversarial results

The independent test
`test_induction_centered_fixed_geometry_rung592_nonfinite_mask_amendment_review.py`
reports `21 passed`.

The positive fixtures use two and three simultaneously affected arrays and
prove distinct canonical paths, exact set equality, sorted index entries,
correct hashes, counts, first coordinates, shapes, and byte lengths. Negative
fixtures reject:

- the old flat-name collision;
- duplicate paths, a missing mask, and an extra finite-array mask;
- wrong shape, payload length, count, first coordinate, content hash, and mask
  content even when the attacker refreshes the other metadata;
- absolute and `..` paths;
- extra or missing index fields and unsorted entries; and
- any mask artifact under four representative non-nonfinite terminals,
  including a normal scientific result.

A fully finite call passes only with no mask index or directory. A
`nonfinite_observation` terminal with neither is rejected.

## Decision

**APPROVE exact commit `6d779ae45b68ffa4c3e7bdf58963cb7f7c2ed2d2` together with its pinned parent amendments for prospective R592 implementation.**

This approval licenses implementation of the frozen contract only. The
completed producer, managed adapter, dry run, and owner tests still require a
different-agent exact-byte review before any managed GPU enqueue.
