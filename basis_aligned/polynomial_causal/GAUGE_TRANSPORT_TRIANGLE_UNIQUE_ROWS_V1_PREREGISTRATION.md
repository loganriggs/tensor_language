# Gauge-transport triangle unique-row authority v1

## Purpose and boundary

This transaction repairs only the row-provenance blocker in the preliminary
`L8 -> L11 -> L14` gauge-transport triangle.  It freezes 96 basis, 96 response-fit,
and 192 evaluation rows with exactly one row from each source document and no
document shared between roles.  It reads no checkpoint, model, response, logit, or
prior triangle outcome.

This authority authorizes only deterministic row materialization from the pinned
caches.  It does **not** authorize the current triangle runner or any model forward.
The rows become eligible for a future Stage-1 runner only after receipt-last
materialization and a separately frozen, source-closed runner authority with the
missing Stage-0/Stage-1 controls.  No row-selection outcome moves any causal,
semantic, composition, OOD, edit, extraction, removal, current-ship, or whole-model
ledger.

## Frozen parent

The sole parent metadata authority is
`basis_aligned/bilinear_quotient/.rowcache/fineweb_oracle_v2_receipt.json`, SHA256
`815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16`.
It binds FineWeb revision `9bb295ddab0e05d785b879661af7260fed5140fc`, ordered
manifest `ba5e92b0d157f47cc6f8656eb1c37e46b7aac6957be8be68c1596736b98e6f90`,
and the pinned first parquet SHA256
`c84e6941d787b50959521df6d6894a91397c8b2db13f8a9c8fe0f8782872e930`.

Four already materialized CPU row caches are eligible:

| source key | file SHA256 | tensor SHA256 |
|---|---|---|
| `n480_skip80` | `2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496` | `343d92ce07f78572e3233120d3361814c63f69fa76e97e58b62d1d6c8f24497f` |
| `n192_skip7000` | `d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c` | `10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0` |
| `n192_skip11000` | `b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868` | `5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa` |
| `n96_skip1200` | `21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f` | `d6302f327983e8233509e0ad8a05aa84fad88784861a9f8d10575b325be83dda` |

## Selection rule frozen before tensor load

Selection operates only on the parent receipt's ordered provenance records.  For
each pool in order, scan row indices in ascending order and accept the first record
for a document not already assigned to any role.  Stop when the role is full.

- basis: 96 documents from `n480_skip80`;
- fit: 96 further documents from `n480_skip80`;
- evaluation: `n192_skip11000`, then `n192_skip7000`, then `n96_skip1200`,
  stopping at 192 documents.

The frozen metadata known answer is:

- basis: 96 from `n480_skip80`;
- fit: 96 from `n480_skip80`;
- evaluation: 105 from `n192_skip11000`, 79 from `n192_skip7000`, and 8 from
  `n96_skip1200`;
- 384 unique documents total, zero cross-role overlap;
- every selected evaluation `dataset_document_index` is greater than every selected
  fit index.

No tensor value, prefix, token class, response, or model outcome participates in
selection.  There is no fallback pool, retry, or post-load reselection.

## Lifecycle

1. `--freeze-authority` requires the freezer, this preregistration, and its focused
   test to be committed, pushed, and byte-identical to the current commit.  It reads
   only the parent JSON metadata, freezes the complete 384-record selection plan,
   and publishes a create-only authority.  Authority reload validation rebuilds the
   entire object, including cache bindings, permissions, and output paths; a merely
   self-consistent edited hash is insufficient.
2. `--materialize` requires that exact authority, replays source and parent hashes,
   then loads the four CPU cache tensors with hash-before/load/hash-after checks.
3. Both stages hold a nonce-bearing owner lock pinned by device and inode.  Ownership
   is rechecked before every publication.  Replacing or editing the lock is terminal
   and the process neither removes the replacement nor publishes under it.
4. Every artifact publication writes and fsyncs a same-directory private temporary,
   hard-links it create-only to the final path, fsyncs the directory, removes the
   temporary, and fsyncs the directory again.  A partial write is never visible at
   the final path.
5. Materialization gathers exactly the authority-selected rows and publishes rows,
   then manifest.  Immediately before receipt publication it replays the exact live,
   committed, and pushed sources; parent receipt; rebuilt authority; all four cache
   byte and tensor hashes; selected cache rows; rows bytes and tensors; and the exact
   rebuilt manifest.  It then publishes the receipt last and reloads the receipt
   through the same strict terminal replay validator.  The receipt binds
   `failure_absent=true`; its validator requires both that field and the actual
   absence of the failure path.
6. Any ordinary materialization exception publishes a create-only failure while the
   process still proves lock ownership.  It never deletes, overwrites, or retries an
   authority, row artifact, manifest, receipt, or failure.  An owner-lock violation
   publishes nothing further and leaves the foreign lock untouched.  Once the
   receipt hard link exists, any later validation, directory-fsync, temporary-cleanup,
   or lock-release exception publishes no failure: receipt-last success and terminal
   failure are mutually exclusive.

The row tensors retain all 513 cached tokens.  A later audited triangle runner may
consume only its registered prefix, but it may not change document allocation.
