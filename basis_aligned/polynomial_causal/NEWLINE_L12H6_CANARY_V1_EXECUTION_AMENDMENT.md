# Newline L12H6 canary v1 execution amendment

Status: prospectively frozen, outcome-blind, and nonauthorizing. This amendment and
its pure CPU sources do not authorize row publication, checkpoint/model loading, a
GPU forward, FINAL/OOD opening, or scientific publication.

## Fresh roles

There is no FIT role. A future model-free transaction must freeze all three roles
before any newline model forward, in the fixed order `CANARY_SELECT`, `FINAL`, `OOD`.
There is exactly one 257-token row per source document and source file. The exact
domain quotas are:

| role | prose | code | list/table | total |
|---|---:|---:|---:|---:|
| CANARY_SELECT | 48 | 48 | 48 | 144 |
| FINAL | 64 | 64 | 64 | 192 |
| OOD | 64 | 64 | 64 | 192 |

This resolves the preregistration's incompatible 96-document recommendation and
128-newline-document minimum without weakening the latter. Every role must have at
least 128 newline-bearing documents and 256 scored newline positions. Selection is
the lexicographic minimum of SHA256 over the frozen allocation seed, role, domain,
document identity, source index, and exact row hash. A row failing its score-only
mask construction is ineligible before ordering. Roles are disjoint in exact source
document, source file, source blob, row, and 32-token prefix. Recursive historical
registry exclusion and source/license identity remain mandatory.

Each candidate also carries an outcome-blind exact role license. Allocation may draw
it only into that role; it cannot spill a short CANARY_SELECT partition into FINAL or
OOD. The separately audited candidate builder must assign role licenses and distinct
structural partitions before row outcomes exist.

`prose`, `code`, and `list` are source-builder metadata, not decoded post-tokenization
guesses. Their exact source partitions, classification rules, files/revisions/blobs,
and licenses must be bound by the future candidate authority. OOD uses disjoint
held-out source partitions inside each of those three strata; it is not a random row
split and cannot be replenished after any outcome.

## Token registry and score-only masks

The tokenizer is `tiktoken` GPT-2 with merge-table fingerprint
`0be287937901b1baae837369293dd6f63da1bece9609006e6485b57a3de37335`.
Classification is by exact decoded token bytes and the priority newline,
quote/bracket-only, other ASCII-punctuation-only, then first-nonspace ASCII capital.
The ordered registry hash is
`0621e173f7b58728bd8436727ab076b4de9eb1f75afd25529978cd569c974984`.
The exact newline IDs are `[198, 628, 44320]`; all four class counts and hashes are
source constants in `newline_l12h6_token_registry_v1.py` and must be recomputed from
the pinned encoding before rows are selected.

Positions 0:64 are never scored. Within a document, position-jitter and random cells
are exactly count-matched to newline targets and exclude newline, punctuation,
capitalization, and quote/bracket cells. These masks are reducer inputs only. No mask,
token class, decoded label, TopK table, or router may reach an execution callback.

## Canary execution and inference

Only `CANARY_SELECT` may be forwarded in the first transaction. Arm order is exactly
`native`, `exact`, `remove`, `head_label_control`. All three nonnative arms replace
only attention site 12 with the already frozen dense nine-head program. `remove`
sets only H6's constant head coefficient to zero; `head_label_control` sets only H7's
coefficient to zero. Native site-12 attention calls are zero in candidates and one in
native; every other attention and all 18 MLP sites execute natively once. All arms
have the exact registered 7,962,698-value dense price. There is no TopK/router arm.

Per-document sufficient statistics include count, CE sum, native-to-arm KL sum, and
correct count for newline, jitter, matched-random, punctuation, capitalization,
quote/bracket, and global-off-target cells. Raw logits, tokens, masks, and target IDs
are forbidden from result publication. The one-sided simultaneous family uses
20,000 source-document bootstrap draws, seed string
`newline_l12h6_canary_v1_bootstrap_20260830`, and all registered canary coordinates
from `newline_coordinate_specs(NewlineScope.CANARY)`. Exact replay and zero-native
physical ledgers are integrity conjunctions, not scientific coordinates.

The original gates remain unchanged: positive simultaneous newline damage and
target-minus-jitter specificity, global collateral UCB at most 0.01 nat and at most
10% of target damage, and positive removal-minus-H7-control lower bound. Exact must
match native write/logits under production currency. Any singular stake, nonfinite
metric, support loss, source drift, or call mismatch is terminal unevaluable.

## Preserved launch blockers

The new sources close tokenizer classification and deterministic fresh-role
allocation only. Launch remains NO-GO until separate committed sources provide:

1. an audited candidate enumerator for pinned prose, code, and list/table sources;
2. a create-only, receipt-last row publisher with recursive registry replay;
3. an external independent row/source audit and pre-forward authority;
4. exact checkpoint-derived program manifests made before any canary forward; and
5. a concrete facade transaction that terminally binds full call closures, sufficient
   statistics, model/input immutability, failure exclusivity, and receipt-last replay.

`FINAL` and `OOD` stay sealed even after row freezing. Their release requires a later
source-bound authority after the canary decision and cannot reuse CANARY_SELECT.
