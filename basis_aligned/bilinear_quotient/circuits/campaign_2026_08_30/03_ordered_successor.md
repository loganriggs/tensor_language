# Ordered successor

## CURRENT tier: 2

L8H7 is a selective causal owner with a weights-readable successor map, but upstream
writers have not been causally attributed.

## Behavior and tensor program

Endpoint: CE and successor-minus-self logit margin after members of ordered families.
Controls use the same tokens outside successor contexts and frequency-matched ordinary
tokens.

Fixed tensor form: L8H7 OV successor map.  Router: a low-rank QK membership/position
selector that decides when the fixed map applies.  Extraction installs the QK/OV
program; removal deletes only its selected successor contribution.

## Evidence

- [`succ_map.py`](../../succ_map.py) and
  [`succ_map_results.json`](../../succ_map_results.json): successor ranks first for all
  eight digit offsets.
- [`succ_general.py`](../../succ_general.py) and result: `0.1478` target damage versus
  `0.00267` elsewhere.
- [`succ_twin_scale.py`](../../succ_twin_scale.py) and
  [`year_succ.py`](../../year_succ.py): weekdays/months/years transport; L14H4 is mostly
  dormant behaviorally.

## Terminal gates

Collateral covers self prediction, copying, numeric formatting, punctuation, and global
CE.  Select on digits; freeze weekdays/months/years and held-out cycle members as OOD.
Default gates apply, with off-target upper bound `0.01` nat.

Shared-owner caveat: all ordered families are one circuit; numeric formatting explicitly
excludes these cells.

**Next experiment:** first-order L8H7 query/key/value writer census with matched
same-layer controls and exact QK/OV replay.
