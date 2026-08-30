# Quote parity and closure

## CURRENT tier: 2

Close-quote prediction has a surgical owner and matched-opener destination, but the
proposed parity-state direction failed causally.

## Behavior and tensor program

Endpoint: close-quote CE under odd unmatched-quote parity.  Controls are opening quotes,
balanced positions, bracket closers, and nearby nonmatching quote tokens.

Fixed tensor form: L13H8/L10H6 matched-opener double-bilinear QK and quote OV payload.
Router: an as-yet-unidentified causal parity carrier must gate the closer program.  A
decoded parity label is not an executable router.  Extraction requires carrier,
matcher, and payload; removal deletes only the parity-conditioned closer contribution.

## Evidence

- [`quote_close_heads.py`](../../quote_close_heads.py): L13H8 target damage `0.5240`,
  about `0.003` elsewhere.
- [`quote_destination.py`](../../quote_destination.py): recent-quote share `0.0695`
  target/`0.0125` control.
- [`quote_state.py`](../../quote_state.py) decodes parity, but
  [`quote_state_causal_results.json`](../../quote_state_causal_results.json) reports only
  `0.0049` gap loss on removal.
- [`quote_mechanism.py`](../../quote_mechanism.py): QK instrumentation and negative
  weights-only controls.

## Terminal gates

OOD splits dialogue/prose, nesting, opener distance, and quote form.  Collateral covers
brackets, punctuation, and newline.  Default gates apply; causally unnecessary decoded
state receives no extraction credit.

Shared-owner caveat: L13H8 is shared with brackets; require quote-only, bracket-only,
and joint cells.

**Next experiment:** fit-only causal parity-state search with matched random subspaces,
frozen before testing whether it feeds the matched-opener head.
