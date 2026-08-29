# Block-3 Family-F v2 reporting recovery

## Scope

Family-F v1 completed its exact fit and reporting call schedule and wrote an exact
program artifact, then failed before result/receipt publication because its terminal
validator compared a CUDA float32 reduction maximum with a CPU maximum.  V1 is spent.
It will not be modified, rerun in place, or given a retrospective receipt.

V2 is a reporting-only recovery.  It does not refit scores, supports, decoders or
affine corrections.  It may publish fit-role reporting metrics only after independently
reconstructing every v1 program tensor from sealed parents.

## Frozen v1 inputs

The v2 authority must be published before deserializing any of these files and bind:

| object | SHA256 |
|---|---|
| v1 authority | `70a4f751d6f79438263eb44f235b24b14334527aca4c169afa77dca6fc701e7d` |
| v1 program | `d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038` |
| v1 failure | `1bb45f2645576fadef564562ef37f98abfb64afb75af8396b882fe63b783f79b` |

The v1 result and receipt paths must remain absent.  The complete exact v1 call ledger
inside the failure must replay.  All original parent, row and checkpoint bindings remain
canonical and are reverified before and after reporting.

## Recovery computation

1. Publish a create-only v2 authority from metadata and hashes.
2. Load v1 authority/failure/program with hash-before/load/hash-after guards.
3. Load the sealed native-gate sufficient statistics and Family-A program.
4. Load the frozen bilin18 checkpoint and reconstruct every v1 program from scores,
   supports, affine parameters and native weights.
5. Require exact tensor equality between reconstructed and saved programs.
6. On the 480 original fit rows, rerun the one native teacher and all 18 frozen student
   arms per logical batch.  No optimizer, support selection or decoder refit is allowed.
7. Require exactly 60 shared prefixes, 60 teacher suffixes and 60 suffixes per student
   arm; sites 0–3 execute 60 times and sites 4–17 execute 1,140 times.
8. Report document-balanced teacher KL, row-mean teacher KL and summed-write NRMSE.
9. Replay direct/polarized equality independently on CPU and CUDA.  Both relative
   residuals must pass `2e-5`; their backend-dependent maxima are not compared.
10. Publish result, reload and semantically reconstruct it, then publish receipt last.

## Permissions and interpretation

- Fit rows only.  Validation and final loaders are forbidden.
- No ground-truth next-token target is used.
- V1 score/affine optimizer traces were never published.  V2 records them as
  `unavailable_from_spent_v1_nonpromotive`; they cannot support convergence claims.
- V2 may recover the original fit comparison because the programs and reporting arms
  were fixed before the v1 outcome.  It cannot earn validation, final, global ledger,
  OOD, extraction or removal credit.
- Only uncalibrated `real_F_post_refit_k256` and `real_F_post_refit_k512` remain possible
  candidates for a separately authorized validation transaction, and only if the
  original registered fit gates pass.  V2 itself opens no such transaction.

## Failure rule

Any mismatch writes a create-only v2 failure and no receipt.  V1 artifacts are never
deleted or overwritten.  No automatic retry is authorized.
