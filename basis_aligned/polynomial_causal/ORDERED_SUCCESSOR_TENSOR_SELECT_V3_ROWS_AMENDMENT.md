# Ordered-successor tensor SELECT v3 row-budget amendment

Status: **prospective NO-GO**.  This amendment is outcome-blind and model-free.  It
does not authorize a row freezer, row access, checkpoint access, model execution, GPU
work, an execution authority, or a scientific claim.  V2 remains an immutable spent
failure and must not be edited, deleted, retried, or reinterpreted.

## Immutable v2 parent

- audited v2 source commit:
  `320dc5537d3fe99b14c29d54f74073714edb21af`;
- v2 independent-audit SHA-256:
  `d5747b99a2ab224fad569c460b5bcc59a695e93ad4595bd300f99e7930d14b1d`;
- terminal v2 failure commit:
  `5f40025895ba0887ad00cc7b3200fc51c8ab823b`;
- terminal v2 failure SHA-256:
  `ba852204a585592c699b8df3554e1dcbe951f964d6da13bbc6d39dbf507278d9`;
- exact failure: `RuntimeError`, `powered successor support requires more than 192
  documents`, `cache_exists=false`, `outcome_access=false`, and
  `terminal_failure_no_receipt`.

Any future v3 owner must stably read and hash the exact v2 audit and failure bytes,
semantically replay the exact failure object, preserve the v2 namespace, and bind both
parents in its own source closure and receipt.  It must use a new cache, lock, audit,
failure, manifest, and receipt namespace.

## Model-free support diagnosis

The exact v2 source, pinned FineWeb parquet, GPT-2 tokenizer, recursive historical
exclusions, decimal lexicon, scored positions `64:256`, and frozen mask builder were
replayed without a checkpoint, model, GPU, or scientific outcome.  The census covered
4,096 collision-free candidate documents from the unchanged start index 200,000.

Under the **registered support-first-then-earliest-unused algorithm**, all three
powered cells first pass after **335 selected documents**, at candidate scan ordinal
1164.  The selected support at that exact stopping point is:

| cell | positions | documents |
|---|---:|---:|
| `positive_clean` | 200 | 136 |
| `wrong_source_clean` | 303 | 189 |
| `no_source_clean` | 234 | 228 |
| `successor_copy_overlap` | 181 | 68 |
| `copy_only` | 328 | 144 |
| `excluded_local_or_ambiguous` | 11 | 10 |

The exact ordered-pair position/document occupancies are: `0->1` 272/141,
`1->2` 219/144, `2->3` 160/106, `3->4` 129/81, `4->5` 132/103,
`5->6` 94/69, `6->7` 86/65, `7->8` 99/73, and `8->9` 66/53.  These
sum to the 1,257 eligible positions.

At the 192nd support-first selection the three primary position counts are only
82, 171, and 137, respectively, although their document counts are 51, 109, and 132.
Thus the v2 failure is a row-budget failure, not an absent-support failure.

For clarity, an exact binary MILP over the same frozen candidate masks has an
unconstrained optimum of 171 documents (200/203/200 primary positions and
144/135/171 primary documents; zero MIP gap).  That is not the preregistered selection
rule and cannot be substituted after observing support.  It is diagnostic only.

Measured CPU wall time for the source/exclusion/harvest/mask/greedy diagnostic was
20.732322 seconds: 5.962899 exclusion load, 9.613328 candidate harvest, 3.358313 mask
construction, and 1.797780 analysis.  A separate exact MILP diagnostic completed in
13.717833 seconds including its repeated source preparation.  Runtime is descriptive,
not a gate.

## Sole prospective v3 scientific change

V3 changes only the SELECT document budget from 192 to **384**.  This is the smallest
multiple of the original 192-document role above the exact registered stopping count
335.  It leaves 49 documents, or 14.63%, of deterministic row-budget margin.  The
support-first prefix is unchanged; after it passes, the role is filled with the
earliest unused candidates exactly as v2 specified.  On the frozen candidates the
384-row census differs from the 335-row stopping census only by one additional
`successor_copy_overlap`, one `copy_only`, one `1->2`, and one `6->7` position.

Everything else remains identical: SELECT role only; one row per unique source
document; start index 200,000; 4,096 collision-free candidates; row length 257;
32-token historical-prefix exclusion; scored positions `64:256`; exact decimal
lexicon; 15 arms; all tensor formulas and rank/null programs; three 200-position and
30-document support gates; bootstrap seed/draws/order statistic; selection gates; and
lowest-price rule.  The larger role has 73,728 scored positions, exactly twice v2's
planned 36,864, and therefore doubles subsequent SELECT forward work.

No global optimizer, reranking, randomized search, threshold change, pair balancing,
or support-dependent budget choice is licensed.

## Remaining launch gates

The accompanying pure CPU module owns only the budget allocator/diagnostic and parent
lineage check.  A future source-closed v3 freezer must still be written, independently
audited at an immutable pushed commit, and run create-only.  It must preserve v2, bind
this amendment and its tests, perform full installed-row/manifest semantic replay,
and publish its receipt last.  Until that separate audit is GO, v3 row materialization
and every model forward remain forbidden.

