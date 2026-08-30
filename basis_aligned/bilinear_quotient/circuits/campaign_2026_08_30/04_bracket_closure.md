# Matched bracket closure

## CURRENT tier: 4

L13H8 is causally surgical and has exact writer-pair score algebra.  That algebra is
dense and not recursively reduced to embeddings.

## Behavior and tensor program

Endpoint: closer-token CE when an unmatched compatible opener is within 64 positions.
Controls kill the nearest incompatible delimiter, previous token, random source, or use
no unmatched opener.

Fixed tensor form: two bilinear QK factors over query/key writer parts plus a closer OV
payload.  Router: compatible-opener identity and distance determine the selected match
edge.  Extraction rebuilds L13H8 from registered match terms; removal kills only the
true match edge.

## Evidence

- [`bracket_match.py`](../../bracket_match.py) and
  [`bracket_match_results.json`](../../bracket_match_results.json): L13H8 deletion
  `+0.8254` target/`+0.00376` global; true-match deletion `+0.6890`.
- [`bracket_query_rank.py`](../../bracket_query_rank.py) and result: rank/random controls.
- [`bracket_pointer_pairs.py`](../../bracket_pointer_pairs.py) and result: exact dense
  writer-pair replay.
- [`bracket_nested.py`](../../bracket_nested.py): nesting controls.

## Terminal gates

Collateral includes incompatible delimiters, quotes, punctuation, and global CE.  OOD
holds out delimiter type, nesting depth, distance, and code/prose.  Default gates apply
and every delimiter subtype must retain effect sign.

Shared-owner caveat: L13H8 also serves quote closure; require bracket-only, quote-only,
and joint-owner cells.

**Next experiment:** recursively reduce the sufficient matched-opener writer-pair
program to embeddings and exact upstream programs.
