# Newline and structural boundary

## CURRENT tier: 4

A selective five-head removal set and exact writer-pair algebra exist, but the known
compressed terms are not newline-specific.

## Behavior and tensor program

Endpoint: newline-token CE.  Controls are position-jitter, count-matched random, other
punctuation, and all remaining positions.

Fixed tensor form: five heads' double-bilinear QK scores and OV payloads.  Router: each
native QK score selects source positions; no external newline-label router is allowed.
Extraction retains the crew plus a frozen writer-pair reconstruction rank; removal
deletes all five.

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

**Next experiment:** recursively test whether a shared tensor grammar beats the known
dense 200-pair requirement on fresh target/control roles.
