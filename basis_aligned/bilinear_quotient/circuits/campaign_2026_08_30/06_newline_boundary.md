# Newline and structural boundary

## CURRENT tier: 4

A selective five-head removal set and exact writer-pair algebra exist, but the known
compressed terms are not newline-specific.

## Behavior and tensor program

Endpoint: newline-token CE.  Controls are position-jitter, count-matched random, other
punctuation, and all remaining positions.

Fixed tensor form: five heads' complete double-bilinear QK scores and OV payloads.
Each native continuous QK score weights source positions; no external newline-label
router is allowed. The known writer-pair compression is nonspecific, so the current
extraction retains the exact stored heads rather than claiming a compressed TopK
program. Removal uses constant global head-index projectors.

## Evidence

- [`newline_crew_screen.py`](../../newline_crew_screen.py) and result identify
  `{7.2,8.2,10.2,11.0,12.6}`.
- [`newline_crews.py`](../../newline_crews.py): `0.6166` target versus `0.0049` elsewhere.
- [`newline_head_pairs.py`](../../newline_head_pairs.py) and
  [`newline_head_rebuild.py`](../../newline_head_rebuild.py): exact score algebra; top
  200/625 retention `0.918` target versus `0.933` control.

## Terminal gates

Collateral includes punctuation, capitalization, quotes/brackets, and global CE.  OOD
splits prose, code, lists/tables, line length, and domain.  A compressed program must
beat equal-rank random writer-pair controls on target-minus-control retention in
addition to default gates.

Shared-owner caveat: late structural heads/writers overlap punctuation and
capitalization services.

## 2026-08-30 source-closed update

The outcome-blind scaffold now freezes an L12H6 canary followed by the full crew
`{L7H2,L8H2,L10H2,L11H0,L12H6}`. Arms are native, exact stored reconstruction,
constant global removal, and fixed +1 head-label control. Evaluation masks cover
newline, jitter, count-matched random, punctuation, capitalization, and quote cells;
tests prove token IDs cannot route execution. The canary stores 7,962,698 values and
the five exact head programs store 39,813,490; candidate calls at replaced sites are
zero-native by contract. Forty focused/shared tests pass.

This is infrastructure, not a new numerical result, and does not raise the tier.
Launch remains NO-GO until fresh disjoint CANARY_SELECT/FINAL/OOD rows, tokenizer-ID
and program authority, terminal owner/publication, and independent review exist.

**Next experiment:** source-close and run the cheap L12H6 canary. If identity, call,
target-removal, and collateral gates pass, freeze the five-head exact terminal without
retesting the failed adaptive writer TopK.
