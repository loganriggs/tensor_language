# Ordered-successor SELECT v2 row-freezer audit: NO-GO

Status: **NO-GO** for prospective row-freezer source commit
`6fc05f07ad08d3a1d129bdfe517755074cbf304d`.

This is a static, non-authorizing audit record. It is deliberately not named
`ordered_successor_tensor_select_v2_rows_independent_audit.json`, does not use the
exact GO schema accepted by the freezer, and cannot authorize row materialization.

The audit was outcome-blind and model-free. The freezer was not run. No row tensor,
checkpoint, model, GPU path, authority, or scientific outcome was opened. Review used
only immutable Git source blobs, JSON metadata, pure tokenizer/mask known answers,
and synthetic CPU tests.

## Immutable identities reviewed

- Source commit: `6fc05f07ad08d3a1d129bdfe517755074cbf304d`
- Amendment SHA-256: `dd09e0b13c48dd2982f1282eb881207e9cd1d08168bdb211adaef5ba412b89f4`
- Freezer SHA-256: `4ef64c5e8f48b2f350692885bfc941b7d32a0824bad03819afb267ec2fd0e748`
- Digit-registry SHA-256: `0a6e9832d1814e38039b3c56b3efb402e0c6a74bcc6c7f837fd22cd68e7f386b`
- Protocol-registry SHA-256: `dfcb82e7d65177c2fbc3136472b05668e975810fcbb5b6dc215397f1c2a88fa8`
- Freezer-test SHA-256: `53cda049d7428b2eae0d2bb5ddf1d44f932d2a39eab5733bbc51e4f0353789f9`
- Digit-test SHA-256: `252cb83232c50dd16a635bf112832b140865a678de05c0173c081213d38c430a`

## What passed inspection

These positive components do not override the blockers below.

1. The decimal lexicon is exactly the non-cyclic order `0,...,9`. Bare and
   one-leading-space surfaces are checked with `encode_ordinary`, exact one-token
   length, exact decode, the frozen token IDs, GPT-2 vocabulary size, and merge-table
   fingerprint.
2. The protocol registry projects the v1 arm names to the intended 15-name list by
   removing only `CURRENT_ONLY` and `V1_ONLY`; the twelve true/null rank arms and
   three baselines are ordered and hash-bound.
3. The pure allocator is deterministic: it scans candidates in canonical order,
   greedily selects a document only while it advances an underpowered primary cell,
   fails if powering needs more than 192 documents, and fills with the earliest
   unused candidates only after every primary threshold passes.
4. The selected role is one row per canonical FineWeb source document. The inherited
   harvester and disjointness validator enforce registered document ID, dataset index,
   exact row, and 32-token-prefix exclusion, and the proposed start index is required
   to exceed every observed historical dataset index.
5. The source identity checks the canonical ordered FineWeb authority, pinned parquet
   size/hash, GPT-2 fingerprint, and digit/protocol registry hashes. The prospective
   amendment and independent-audit gate remain non-self-authorizing.
6. Every successor mask cell is partition-validated. The three powered cells are
   censused against 200 positions and 30 documents; all named copy/overlap/exclusion
   masks and pair-index tensors are content-hashed.
7. Staged row and manifest files are fsynced and semantically reloaded, installed
   files are hash-before/load/hash-after validated, and allocation is independently
   re-harvested and recomputed before terminal publication.
8. The inode/nonce lock, terminal-absence checks, create-only hard links, and
   receipt/failure mutual-exclusion guards are otherwise coherent.

## Blocking defects

### B1. The declared source closure omits direct and transitive scientific code

`SOURCE_PATHS` contains only 17 paths and omits modules whose behavior directly
defines masks, arms, scorer currency, and imported execution types. At minimum the
following paths listed by the reused v1 statistics closure are absent:

- `ORDERED_SUCCESSOR_TENSOR_DISCOVERY_PREREGISTRATION.md`
- `bilin18_observed_model_facade.py`
- `circuit_campaign_runtime.py`
- `circuit_successor_tensor.py`
- `ordered_successor_masks_v1.py`
- `ordered_successor_tensor_discovery_v1.py`
- `ordered_successor_tensor_backend_adapter_v1.py`
- `successor_attention_backend.py`
- `tensor_preserving_attention.py`
- their four declared focused test files

The corresponding facade/model implementation dependencies are also transitive.
The current source-closure test checks only five v2 files plus the inherited row-base
closure, so it cannot detect these omissions.

Required repair: either make the v2 registry/freezer genuinely standalone and
model-free, or include the complete transitively executed source/test/preregistration
closure. Add a test that derives the required closure from imports/protocol sources
and requires exact set equality, rather than a small `issubset` smoke check.

### B2. Importing the freezer contradicts its model-free receipt claim

Importing `prepare_ordered_successor_tensor_select_v2_rows` immediately places all of
the following in `sys.modules`:

```text
ordered_successor_tensor_discovery_v1
circuit_campaign_runtime
bilin18_observed_model_facade
jacclust.tt_model
successor_attention_backend
```

This occurs because the protocol imports v1 discovery and the statistics module
imports discovery/backend code. The amendment says the freezer imports no model
loader, while the proposed receipt would assert `model_imported=false`.

Required repair: move the immutable arm/candidate/statistical constants needed by the
row freezer into a pure CPU/data-free v2 protocol object. The freezer must not import
the observed facade, model implementation, execution runtime, or physical factor
backend. Add an import-surface test against a fresh Python process.

### B3. The 15-arm protocol is incompatible with the named unchanged scorer

The protocol binds 15 arm names, but
`ordered_successor_tensor_select_statistics_v1.SelectDocumentLedger` and downstream
scoring/ruling functions hard-code `discovery.ARM_NAMES`, which has 17 arms. A pure
known answer with metric tensors shaped `[document,15,cell]` is rejected as
`successor metric ledger is malformed`. The existing v2 tests compare arm-name lists
but never pass a 15-arm ledger through the registered scorer.

Required repair: define and source-bind a v2 scorer currency that consumes exactly the
15 canonical arms while preserving the registered formulas, draws, seed, order index,
thresholds, gates, and lowest-price rule. Add a complete 15-arm ledger/scorer known
answer and reject 14-, 16-, and 17-arm inputs. No zero-filled omitted diagnostic may
be used to retain the old 17-arm shape.

### B4. Required descriptive pair occupancy is not published

The amendment says pair occupancy remains a mandatory descriptive output. The
manifest stores a hash of `pair_index`, but `powered_census()` reports only named mask
cells. It does not publish per ordered digit pair position/document counts. A hash is
not the registered descriptive occupancy report used by the v1 scoring currency.

Required repair: publish deterministic position and document occupancy for every
non-cyclic pair `0->1,...,8->9`, bind exact pair order/names, recompute it from the
installed rows during semantic replay, and test count closure against eligible-cell
occupancy.

### B5. Installing the freezer's own manifest makes final protected replay fail

This is a deterministic liveness failure. The freezer snapshots
`base.discover_registry_files()` before cache installation. That function recursively
discovers every `*manifest*.json`. The cache installation then creates
`CACHE/select_manifest.json`, which matches that pattern. In `final_guard`,
`_protected_replay()` calls `base.verify_snapshot()`, which rediscovers registries and
requires exact membership equality with the pre-install tuple. The newly installed
manifest therefore changes membership, so receipt publication must fail. The only
possible real transaction at this source is an installed cache followed by terminal
failure; it cannot produce a receipt.

Required repair: use a local registry census with one exact, source-bound exclusion
for the freezer's own in-progress manifest (while validating that manifest separately),
or extend the inherited verifier with an explicit immutable self-artifact exclusion.
Add a production-shaped test that takes the pre-install census, creates the installed
manifest, and proves final protected replay still sees exactly the intended historical
registry universe and no broader omission.

### B6. A post-link receipt durability error still propagates as transaction failure

`_write_json_create_only()` semantically reloads the temporary JSON and hard-links it
create-only, but calls directory fsync after the link without terminal-state handling.
If that fsync raises, the receipt exists, failure publication is correctly suppressed,
yet `freeze()` raises. This violates a receipt-last lifecycle in which an exact linked
receipt cannot be converted back into a reported failed transaction.

Required repair: after the guarded hard link succeeds, make all directory-fsync and
temporary-cleanup errors non-propagating terminal durability warnings, or exact-reload
the linked receipt and return terminal success. Add injected post-link-fsync and
cleanup regressions proving that no contradictory failure is created and the caller
observes the receipt terminal state.

## Reproduced tests and counterexamples

The immutable source was extracted rather than imported from the moving shared tree:

```bash
succ_tmp=$(mktemp -d)
git archive 6fc05f07 basis_aligned/polynomial_causal jacclust | tar -x -C "$succ_tmp"
PYTHONPATH="$succ_tmp/basis_aligned/polynomial_causal:$succ_tmp" pytest -q \
  "$succ_tmp/basis_aligned/polynomial_causal/test_ordered_successor_digit_lexicon_v2.py" \
  "$succ_tmp/basis_aligned/polynomial_causal/test_prepare_ordered_successor_tensor_select_v2_rows.py" \
  "$succ_tmp/basis_aligned/polynomial_causal/test_ordered_successor_tensor_select_statistics_v1.py" \
  "$succ_tmp/basis_aligned/polynomial_causal/test_ordered_successor_masks_v1.py"
```

Result: **25 passed in 3.61 seconds**. These tests establish the positive components
above but do not close B1-B6.

The outcome-blind import/shape harnesses produced:

```text
declared_source_count 17
ordered_successor_masks_v1.py bound False
ordered_successor_tensor_discovery_v1.py bound False
circuit_campaign_runtime.py bound False
bilin18_observed_model_facade.py bound False
successor_attention_backend.py bound False
fifteen_arm_ledger 15 REJECTED successor metric ledger is malformed
successor_manifest_would_match *manifest*.json True
```

No exact independent GO audit was created.

Reviewer: Codex independent outcome-blind reviewer.
