# Early-MLP/context cross v1 implementation amendment

**Frozen before any model outcome:** 2026-08-28

**Status:** implementation contract only; GPU/model execution remains **NO-GO** until
every source named below is committed, pushed, tested, independently audited, and a
fresh pre-outcome authority binds that exact future commit.

This amendment closes the operational choices intentionally deferred by
`EARLY_MLP_CONTEXT_CROSS_V1_PREREGISTRATION.md`. It does not change any mask, pivot,
cell split, statistical gate, or claim boundary in that document.

## Exact inherited inputs

### Rows

The only licensed evaluation tensors are:

| role | serialized file SHA256 | raw tensor SHA256 | rows | source documents |
|---|---|---|---:|---:|
| `skip7000` | `d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c` | `10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0` | 192 | 79 |
| `skip11000` | `b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868` | `5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa` | 192 | 105 |

The canonical row/provenance receipt file SHA256 is
`815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16`.
The loader must verify the serialized hash before load, the serialized hash again
after load, the raw tensor hash, ordered row/provenance bindings, and exactly zero
source-document overlap between the two roles. A mismatch occurs before model load
and writes only a create-only failure receipt.

### Model and shared program

- config SHA256: `428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`
- weights SHA256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`
- implementation SHA256: `94b927b7d0576b29f2bfd4dbee851462598ae944d3c2f6d9405e702163fdbc4d`
- model realization SHA256: `cf3ca3f55028979ef6f87ac4afa08a7d90fc01dfa4fc2ce037343ac3c69688eb`
- component-tree SHA256: `94cbebb35ca3f8c6923f5040b76d243c3f3fa192496604bd40abeb2e4077da0c`
- inherited backend source SHA256:
  `738f4988fe5b87a7329f833bc7117cc417adcb9834da06533b00bc8b320c18e0`
- immutable shared-program content SHA256:
  `cad513c942cccaf01e747cb600428b427c03d98dd0dddc710a4028ff1ba9d0bb`

The program may be deterministically reconstructed once with the parent backend's
production loader and `_build_program`; a new serialized program artifact is not
required. The new backend must content-hash the reconstructed `SharedProgram` before
the first outcome and after the last outcome and require the exact content hash
above. The old program-bank hash `a974...` is forbidden because it binds old mask
descriptors, not merely the reusable shared rows.

The physical family remains centered rank-64 covered tables plus the executed
rank-64 embedding-to-row prediction for uncovered tokens. Output-nearest-neighbour
indices remain a hashed control and are never described as the executed fallback.

## Required source closure

The launch source closure contains the exact committed bytes of:

1. the preregistration, this amendment, registry, and their tests;
2. `early_mlp_context_cross_v1_statistics.py` and its tests;
3. a new role-scoped measurement request/collector module and its tests;
4. a thin bilin18 backend and its tests;
5. a create-only two-role transaction/lifecycle runner and its tests;
6. the inherited observed-model facade, model implementation, and reusable parent
   backend helpers; and
7. the launch command source itself.

The authority must bind the future implementation commit and require that commit to
equal `HEAD`, be an ancestor of `origin/main`, and have no dirty tracked or untracked
file inside the closure. No current commit hash is pre-authorized.

## Measurement transaction

Acquire one fresh exclusive lock and require a completely absent output namespace.
Then, in order:

1. verify all frozen inputs and the source closure;
2. load and prove the two role identities/disjointness;
3. load and hash the model; reconstruct and hash one shared program;
4. publish a create-only, outcome-blind authority;
5. execute `skip7000`, then `skip11000`; within each role execute cells 0 through 63
   in registry order;
6. for cell `(0,0)`, install no substitution hooks; all 36 native modules execute;
7. for every other cell, native modules execute first and exactly the registered
   module outputs are substituted from the same immutable bank;
8. retain per-row correct counts, float64 CE sums, and scored-token counts, then
   aggregate in literal row order to source documents;
9. verify model tree, component tree, program content, inputs, lock, call ledgers,
   and source closure again;
10. publish one two-role tensor payload, then its manifest, and publish the terminal
    tensor-free receipt last.

No cell statistic, partial role, development score, or validation outcome is
published before both roles and all 64 cells close. Any exception publishes a
create-only failure receipt containing hashes/counts and an error class/message but
no logits, row tokens, CE columns, or other partial outcome.

The request/collector cannot reuse the old cut-rank `MeasurementRequest`, request
plan, collector, descriptors, or call ledger: those encode always-compiled attention0
and MLP0. New requests derive only from `mask_for_cell`; `(0,0)` is fully live and
MLP0 is a genuine factor.

## Staged scoring

The terminal payload is split into three sealed per-role capabilities:

- discovery: exactly 48 cells;
- validation: exactly seven cells;
- heldout: exactly nine cells.

Rank three receives discovery to fit and validation only to score; it has no heldout
argument. Rank four receives discovery plus validation to fit and heldout only to
score. Every capability binds role, authority, ordered document identity, document
token counts, exact cells, and tensor hashes. Extra, missing, duplicate, nonfinite,
or mixed-role values fail before scoring.

The bootstrap and ALS algorithms are exactly those in the preregistration. One
source-document multiplicity vector per role/draw is shared across every cell,
target, and baseline. Quantiles use type-7 linear interpolation. The frozen pivot
singularity rule is

\[
s_{\min}\le 10^{-12}\max(s_{\max},1).
\]

Any singular draw, zero interaction-NRE denominator, zero total-cost variance for
\(R^2\), zero additive RMSE for the cross/additive ratio, zero ALS normalization RMS,
or other nonfinite metric is a hard failure for that role/rank. No zero denominator
is replaced by zero, one, epsilon, or an omitted draw.

Each role is scored independently. The two-role scientific conjunction passes only
if both roles' CE conjunctions pass. No pooled role statistic may rescue a failure.

Top-1 remains a mandatory secondary report with the same raw metrics and subgroups,
but this version defines no numerical top-1 pass gates. Therefore it authorizes no
formal “broad behavioral pass” boolean. Only the registered narrow CE factorization
claim may pass; top-1 discrepancies must be reported and can motivate a separately
preregistered broad-behavior gate later.

## Fresh artifact namespace

The implementation must use one namespace never used by an earlier attempt, with
distinct create-only files for pre-outcome authority, terminal payload, manifest,
terminal receipt, and integrity failure. The authority and receipt store full hashes,
never the abbreviated hashes used in prose. A failed or partially existing namespace
is spent and cannot be retried in place.

## Tests required before launch review

- exact 64-cell order; skip/duplicate/mixed-role rejection; `(0,0)` empty mask;
- role support, raw/file hash, provenance, and document-disjointness checks;
- exact stage support plus extra finite, extra NaN, missing, and mutation rejection;
- native-before-substitute execution and 36 native modules at `(0,0)`;
- exact rank-three/rank-four cross recovery and forbidden-stage poison tests;
- ALS known answer, scale equivariance, seeds, 100-sweep count, update order, and tie;
- token-weighted paired bootstrap, type-7 quantiles, and singular/zero-denominator
  retention;
- independent two-role conjunction;
- source closure, protected hashes, lock theft, create-only collision, failure
  publication, and terminal receipt-last ordering.

Passing these tests permits an independent launch-readiness audit. It does not itself
authorize model execution.
