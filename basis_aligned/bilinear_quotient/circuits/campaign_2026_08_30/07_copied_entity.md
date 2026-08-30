# Copied proper-name and entity continuation

## CURRENT tier: 2

The endpoint is strongly affected by copy/capitalized committees, but its conditional
upstream interaction is not yet attributed.

## Behavior and tensor program

Endpoint: CE on capitalized targets with an earlier identical antecedent.  Matched cells
are novel capitalized and copied non-capitalized targets; global text is collateral.

Fixed tensor form: capitalized/name-fragment OV payload.  Router: antecedent-match QK
scores select when and where that payload applies.  Extraction installs only the
interaction-resolved matcher and payload; removal is antecedent-gated.

## Evidence

- [`circuit_capcopy.py`](../../circuit_capcopy.py) and
  [`circuit_capcopy_results.json`](../../circuit_capcopy_results.json): committee13
  damage `1.6060` copied-capitalized, `0.3456` novel-capitalized, `0.7471` copied-
  noncapitalized, `0.0277` global.
- [`circuit_copy2.py`](../../circuit_copy2.py): logit/pattern candidate sets.
- [`capitalized_committee12.py`](../../capitalized_committee12.py): late payload hooks.

## Terminal gates

OOD holds out entity strings, BPE forms, lag, frequency, and domain.  Default gates
apply; copied-capitalized damage must exceed both matched control classes with
simultaneous positive lower bounds.

Shared-owner caveat: this is the interaction of copy and capitalization, not a disjoint
third parameter set.

**Next experiment:** conditional first-order writer/reader census and a
matcher-by-payload factorial.
