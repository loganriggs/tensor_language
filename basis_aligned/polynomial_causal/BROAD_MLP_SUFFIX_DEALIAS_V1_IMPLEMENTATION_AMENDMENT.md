# Broad-MLP suffix de-alias v1 implementation amendment

**Frozen before any MLP-only layers-3--8 outcome:** 2026-08-29

**Status:** implementation contract; NO-GO for canonical execution until every file
below is committed/pushed, focused tests pass, and independent audit returns GO

This amendment operationalizes
`BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md` without changing its masks,
estimands, bootstrap, gates, or claim boundary.

## Frozen parents

- preregistration SHA256:
  `065f1458a93685432596abf43c6628935e93c4e84d8210cb19d7f346d1e9b24f`
- old measurement authority:
  `2bca3a523c3fbcb95b67766033c54d908d00784378dab0216572147780834463`
- old measurement payload:
  `2bf8c5c61038a6b9cc5437357e4ba45373a18f3312e6a80f3c86d1207251d279`
- old measurement manifest:
  `4ca06eb4b2d854469317e487d3dac7a06ef86a1c441e3fe54db03b2505744e9f`
- old terminal measurement receipt:
  `82af48ef6a553038316004dfcf1e82eb10f9d717fde3f8021c805c0afd79da43`
- old score results, descriptive/audit binding only:
  `ed60ea243a6fecff7eecf6a7284059628124e7f3f070c719c3f2aa352f0fd940`
- old measurement source closure:
  `207f9e91ae4d16af293563f65337bb7dfe8666542cd37186cb9ec14b7cd9e437`
- old program bank:
  `10f253d1f89109b864fda7dff6d16b40212326600146d0b429006c017af6e443`
- shared replacement program:
  `cad513c942cccaf01e747cb600428b427c03d98dd0dddc710a4028ff1ba9d0bb`
- model realization:
  `cf3ca3f55028979ef6f87ac4afa08a7d90fc01dfa4fc2ce037343ac3c69688eb`
- component tree:
  `94cbebb35ca3f8c6923f5040b76d243c3f3fa192496604bd40abeb2e4077da0c`

## Exact row/document join

| role | row file | raw rows | ordered row identity | row→document | ordered documents |
|---|---|---|---|---|---|
| skip7000 | `d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c` | `10d66676c804569eaa501d0c3c425f357d1d47924565d8edb62e06fbe37b3a1fa` | `d5dffab188ccccf28fe48170e6e9000b9f57789c7038a1c2d8c232c1684686da` | `0d0c6fea333d3ce70cfb1fb7206fd33f1f91465e01e7f3857c9294b6d90ce0cc` | `b28446ac631be5c27ddb88c94416f9c4bdb79d8382106c31876f0b91f4709f51` |
| skip11000 | `b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868` | `5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa` | `929016f88e2e5ac9798495053a6dd12399238822da2fe8afb3bbdb731f61bf7a` | `3b9931a1d26b15e2691cfa9a4eafeb136e3c2d83a050513c0f0c23bc3e4fc682` | `19583d31e77cd68b73e1657aaaaf57bbd90108fe1edc0674b0867f69d845e1ed` |

Both roles have ordered row-token-count SHA256
`1355494a19d118e5ff051d548120eb448113ba17b3dc828d83a6270e0cfa3bec`.
The implementation must also bind the old stage-statistic hashes and common scored
target/support preimages, not merely the values abbreviated here.

## Frozen registry and execution order

For ordinal $i=0,\ldots,7$, execute exactly

$$
P_i\cup\{\operatorname{MLP}_3,\ldots,\operatorname{MLP}_8\},
$$

in prefix order `0,1,2,3,4,5,6,7`, first on `skip7000`, then on `skip11000`.
Every cell uses 192 rows, 24 batches of eight rows, and 192 scored tokens per row.
Expected totals are 16 cells, 384 outer forwards, and 13,824 native-module calls
(36 native sites × 384 forwards). The 768 figure in the strategic cost comparison
counts role-cell four-row-equivalent batches from a different batching convention;
the canonical backend currency here is 384 eight-row outer forwards. Reports must
name which currency they use.

Native modules execute before exact output substitution. The cell ledger must record
all 36 native calls per batch and substitutions at exactly the registered prefix plus
MLP3--MLP8 sites. No attention substitution, fitter call, retained logits, gain fit,
or mask-specific program is permitted.

## Frozen artifact namespaces and ordering

Measurement namespace:
`broad_mlp_suffix_dealias_v1_measurement_wave_v1`.

Score namespace: `broad_mlp_suffix_dealias_v1_score_v1`.

For measurement, publish create-only authority before outcomes, then one tensor
payload containing both complete roles, then manifest, then terminal receipt. A
failure receipt is terminal and forbids interpretation. No cell, partial role, point
metric, bootstrap metric, or score may be published early.

The score transaction may open new outcomes only after validating the terminal new
receipt and the exact sealed old authority/payload/manifest/receipt chain. It publishes
results then receipt last. Canonical measurement and score require exact default
paths; injected/test paths are structurally nonauthoritative and may not be laundered
into canonical scoring.

## Exact source closure

The canonical source closure contains these files plus every transitive production
backend/model-builder source enumerated by the implementation:

- `BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md`
- `BROAD_MLP_SUFFIX_DEALIAS_V1_IMPLEMENTATION_AMENDMENT.md`
- `broad_mlp_suffix_dealias_v1.py` and its test;
- `broad_mlp_suffix_dealias_v1_measurements.py` and its test;
- `broad_mlp_suffix_dealias_v1_lifecycle.py` and its test;
- `broad_mlp_suffix_dealias_v1_bilin18_backend.py` and its test;
- `run_broad_mlp_suffix_dealias_v1.py` and its test;
- `score_broad_mlp_suffix_dealias_v1.py` and its test;
- all reused early-context-cross lifecycle/scoring/backend sources;
- all reused cut-rank adapter/backend/model-facade sources;
- `jacclust/tt_model.py`.

The launch commit must be pushed and an ancestor of `origin/main`. Every closure file
must match its committed bytes before authority, after each role, before every
publication, and at terminal closure.

## Required focused tests before GO

1. exact eight masks and zero attention substitutions;
2. all 16 role-cell receipts and full native/substitution call census;
3. old/new exact document, denominator, target/support, program, and physical-cell join;
4. known-answer large prefix-invariant raw suffix synergy with zero three-way $Q$;
5. prediction error equals $Q$ exactly;
6. zero-$Q$ descriptive cosine is undefined without failing decision gates;
7. zero NRE/R2 denominator fails closed;
8. paired within-role bootstrap and conditional target-only cross-role bootstrap;
9. type-7 known-answer quantiles and exactly 2,000 draws/seeds;
10. source mutation, model/program mutation, old-artifact mutation, race, duplicate,
    skip, wrong order, partial publication, and receipt-last failures;
11. injected/synthetic measurement cannot receive a canonical score;
12. stale inherited predicate/docstring gate check over every new executable;
13. canonical namespaces are pristine immediately before launch.
