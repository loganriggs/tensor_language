# Gauge-transport triangle unique-row v2 recovery

## Reason and immutable parents

V1 spent its create-only namespace before rows publication.  Its source-cache check
compared the FineWeb parent's `tensor_raw_sha256` (SHA256 of contiguous raw tensor
bytes only) against `tensor_sha256` (dtype, then JSON shape, then raw bytes).  Those
are intentionally different hash currencies, so all four cache checks would fail;
the first deterministic failure was `n192_skip11000`.

V2 preserves these immutable parents:

- v1 authority file SHA256
  `5f7435150561ef385c9a4ee51e2040c4a029e98faefbfe1bc0f92612d820498e`,
  internal authority SHA256
  `8901a7446f70358e7e058013bb81c72f477c8636f5a1f76088307eda437025b5`;
- v1 failure file SHA256
  `91859b52b55b8be8ac05dc61f26b95fd43cdb92db7b8c39dfa72d226df41eb58`;
- absence of v1 rows, manifest, and receipt;
- selection plan SHA256
  `0d66f060a43959c94afc14691b4a19730147c942da94807f919513fb8c421629`.

V2 never deletes, edits, retries, or reinterprets a v1 artifact.

## Sole protocol change

The document allocation, 96/96/192 role sizes, source pool order, exact 384-record
selection plan, cache files, lifecycle, permissions, and nonpromotive claim boundary
are identical to v1.  The sole recovery change is hash-domain correctness:

- source-cache tensors are checked against the parent receipt using
  `SHA256(contiguous CPU raw tensor bytes)`;
- output role tensors retain the v1 composite convention
  `SHA256(dtype UTF-8 || JSON shape || contiguous CPU raw tensor bytes)`.

The authority names both currencies and their domains.  They must never be compared
or substituted.  The v2 authority rebuild requires exact equality to the spent v1
selection plan, not merely equality of counts or its digest.

## Source closure and lifecycle

The v2 preregistration, freezer, focused tests, and reused v1 freezer are exact
committed-and-pushed source dependencies.  V2 uses a separate authority, rows,
manifest, receipt, failure, and owner-lock namespace.  It retains v1's nonce/inode
lock, atomic temp+hard-link create-only publication, directory fsyncs, receipt-last
ordering, terminal replay, and receipt/failure exclusivity.

Freezing the authority reads only JSON metadata and the two v1 JSON parents.
Materialization later loads only the four already pinned CPU caches.  Neither stage
authorizes a model forward, triangle runner, training, causal/semantic/composition
claim, or global ledger credit.
