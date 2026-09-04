# Rung 592 nonfinite-mask naming amendment

Date: 2026-09-04 00:57 UTC  
Status: prospective, outcome-blind, CPU-only specification; independent approval required before implementation  
Diagnostic-prefix amendment commit: `3be7c21c3886502ea989efdaeba5c137aef45d8e`  
Diagnostic-prefix amendment review SHA-256: `e7373c2249e0456327d386559d4f3fa68e0661ed076a35fb120ad9d8effaa675`

## Scope

This amendment closes the sole blocker in the independent review of the diagnostic-prefix amendment. It changes no
scientific object, authority, split, row, site, role, machine arm, formula, threshold, bootstrap, call order, complete
evidence, normal terminal, or forward price. It also changes none of the completed-call, immediate-stop, hard-abort,
per-call-directory, or receipt-last rules.

The parent used the single filename `nonfinite_mask.npy` for every affected raw array in a failing call. Two affected
arrays would therefore collide. Replace only that filename clause with the exact one-to-one scheme below.

## One raw array, one mask

Inside the final failing-call directory, create:

```text
nonfinite_mask_index.json
nonfinite_masks/{raw_stem}.mask.npy
```

Every mandatory raw evidence filename in a call directory is unique and has the form `{raw_stem}.npy`. For each and
only each raw float array containing at least one IEEE NaN or infinity, write exactly one C-contiguous NumPy boolean
array at `nonfinite_masks/{raw_stem}.mask.npy`. It has the same shape and row-major coordinate order as the raw array,
and element $i$ is exactly

$$
m_i=\neg\operatorname{isfinite}(x_i).
$$

For example, simultaneous nonfinite values in `logits.npy` and `hook_deltas.npy` require the distinct files
`nonfinite_masks/logits.mask.npy` and `nonfinite_masks/hook_deltas.mask.npy`. The flat path `nonfinite_mask.npy` is
forbidden. No mask may exist for a finite array, and every nonfinite float array must have one mask.

`nonfinite_mask_index.json` is a JSON array sorted lexicographically by `raw_filename`. Each entry has exactly:

```json
{
  "raw_filename": "hook_deltas.npy",
  "mask_filename": "nonfinite_masks/hook_deltas.mask.npy",
  "raw_dtype": "float32",
  "mask_dtype": "bool",
  "shape": [],
  "mask_byte_length": 0,
  "mask_sha256": "",
  "nonfinite_count": 0,
  "first_lexicographic_coordinate": []
}
```

At runtime, `shape`, `mask_byte_length`, `mask_sha256`, `nonfinite_count`, and
`first_lexicographic_coordinate` contain the observed exact values. `shape` must equal both the raw-array shape and
mask-array shape. `mask_byte_length` must equal the product of the shape dimensions because NumPy bool uses one byte.
`nonfinite_count` must be positive and equal the sum of the mask. `first_lexicographic_coordinate` is the first true
C-order coordinate and must have one integer per array axis. The raw file's own dtype, shape, length, and SHA-256 remain
bound in the parent `call_prefix.jsonl` evidence map.

The index's set of `raw_filename` values must equal exactly the set obtained by applying `not isfinite` to all raw float
arrays in the failing-call directory. Mask filenames must equal the deterministic stem mapping above, be unique, and
remain inside `nonfinite_masks/`. Any duplicate, missing, extra, aliased, absolute, or parent-traversing path is an
integrity failure. Integer, boolean, JSON, and token arrays are not eligible for a nonfinite mask.

If the terminal predicate is not `nonfinite_observation`, neither `nonfinite_mask_index.json` nor the
`nonfinite_masks/` directory may exist. If it is `nonfinite_observation`, both must exist and the index must be
nonempty. As before, nonfinite evidence is legal only in the final failing completed call of an invalid diagnostic and
can never appear in normal scientific evidence.

## Independent review gate

Before implementation resumes, an independent reviewer must extend the frozen diagnostic-prefix tests with at least:

- two and three simultaneously nonfinite raw arrays, proving distinct masks and exact set equality;
- a flat-name collision, duplicate mask path, missing mask, extra finite-array mask, wrong shape, wrong byte length,
  wrong count, wrong first coordinate, and wrong content hash;
- absolute and `..` mask-path traversal; and
- mask files present under any non-`nonfinite_observation` terminal.

Only a committed APPROVE of this exact amendment plus its parent amendments permits R592 implementation work. The
implementation still needs a separate different-agent exact-byte approval before managed GPU enqueue.
