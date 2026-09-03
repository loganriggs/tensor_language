# Rung 521 preflight addendum: feasible same-half controls and coherent donors

**Frozen:** 2026-09-03 05:17 UTC, before any rung521 CUDA execution, smoke result, model outcome, or
optimization.

This addendum corrects two implementation details in
`ATTENTION8_SHARED_PRIVATE_DAS_RUNG521_PREREGISTRATION.md`. It does not change the targets, document splits,
exclusive-cell rule, ranks, seeds, Stage-A thresholds, permutation count, bootstrap count, predictions B--E, or
registered interpretations.

## 1. Why a correction was necessary

The original four-level matched-control rule is not feasible when controls are correctly kept inside each FIT half.
It fails in 8 of the 16 quartet evaluation cells. The failures are small but exact: examples include two
`r.2.0.1` FIT-half-0 members and one `r.2.0.2` FIT-half-0 member with no candidate even at the final
token-class-plus-CE-decile level. `r.2.2.1` on TEST also has a Hall deficit: every member has at least one candidate,
but no one-to-one assignment exists.

Matching once across all FIT rows and then splitting by the member's half would make the code run, but it would allow
a half-0 member to use a half-1 control. That would contaminate the registered half-to-half power test, so that
workaround is forbidden.

The implementation instead uses a deterministic augmenting-path bipartite matcher separately inside each data cell.
All controls remain:

- inside the circuit's parent slice;
- inside the same FIT half, VALIDATION split, or TEST split as the member;
- outside that circuit's own full member mask; and
- outside the complete four-circuit attention8 quartet.

## 2. Corrected control hierarchy

The first four levels are unchanged. Only if they cannot complete a one-to-one matching does the matcher continue:

1. same next token, position bin, and native-CE decile;
2. same next token and native-CE decile;
3. same token class, position bin, and native-CE decile;
4. same token class and native-CE decile;
5. same token class and position bin;
6. same position bin and native-CE decile;
7. same token class;
8. same position bin;
9. same native-CE decile;
10. same parent slice and data cell, with the exclusions above.

The seven token classes are newline/control-line, number, punctuation/symbol, space-initial capitalized word,
space-initial alphabetic word, alphabetic continuation, and mixed/other. Position bins have width 32. Native CE means
the unmodified model's per-token cross-entropy loss; deciles are computed separately inside FIT, VALIDATION, and TEST.

The full preflight constructs 54,014 member/control pairs across four data cells, four quartet-exclusive masks, and
the 32-circuit fingerprint masks. Counts by level are

`2,988, 6,306, 41,440, 2,795, 317, 165, 3, 0, 0, 0`.

Thus 485/54,014 = **0.90%** require any new terminal level, only three pairs cross token class, and no pair uses the
last three broadest fallbacks. For the primary quartet-exclusive controls, 16/1,886 = **0.85%** use a new level:
ten use token-class plus position, six use position plus CE, and none cross token class. Every matching count and map
hash is stored in the frozen preflight receipt.

## 3. Corrected donor construction

A whole-attention8 interchange needs one coherent donor sequence. Independently matching each token position could
splice 256 different documents into one artificial attention write. Therefore one donor map is a permutation of
whole rows:

`donor_write[recipient_row, position] = native_write[donor_row, position]`.

Every donor row is in the same data split, comes from a different document, and is used at the same token position.
The row matcher prefers the same decile of **row-mean native CE**, creates a one-to-one row permutation, and forbids a
recipient from reusing a donor across the eight maps. Four maps form D0 and four form D1. FIT, VALIDATION, and TEST
have independently constructed maps.

The actual data admit the stronger design without relaxation: every one of the 4,544 FIT row assignments and every
one of the 1,728 assignments in each 216-row held-out split matches the row-mean CE decile exactly. The mean absolute
decile distance is zero for all 24 maps. The preflight receipt stores forward and inverse D0/D1 hashes, verifies that
every map is bijective and different-document, and verifies eight distinct donors per recipient.

## 4. Frozen execution price and stop rule

Stage A uses 568 FIT rows in batches of four:

- capture plus independent native replay plus exact self-donor replay: `3 * ceil(568/4) = 426` forwards;
- two four-map ensembles in both swap directions: `16 * ceil(568/4) = 2,272` forwards;
- total: **2,698 inference-only forwards, zero backward calls, and zero learned values**.

The instrument-only smoke precedes this run and retains no task or circuit metric. It freezes a positive attention8
edit-RMS floor. Stage A still stops before constructing an optimizer. If Prediction A fails, no shared/private DAS
claim may be read; the next action is more donors if the ensembles disagree, otherwise more documents.

## 5. Frozen identities

- original preregistration SHA-256:
  `e40ca9654485d8fcc04dd09e0b86628fa633e98d97c0b444c6661f56f73461de`
- full preflight receipt SHA-256:
  `42639d35ef6317104c6e0e684aeb00cb4c550df77d496733bcfe8be790fed650`
- Stage-A executable SHA-256:
  `d5ca962c16cd8f454adac79916a9cf3272b91debac0d27ebba2ce77804fb9ebd`
- shared mathematical/matching library SHA-256:
  `edcf3d750e8fbdcb2ae479bcc6e68bd7ccc5078217b62cf981570656b6a773e4`
- instrument-only smoke executable SHA-256:
  `63fb484ada61a10168a341db8f3134c0dc0faa65f41954d806cb6fff1705e615`

The preflight receipt is
`basis_aligned/bilinear_quotient/attention8_shared_private_das_rung521_preflight.json`. It is the authoritative source
for all 144 control-match hashes and all 24 donor maps.
