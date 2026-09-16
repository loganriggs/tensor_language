# Extracted MLP9 contextual DCT node rank-16 fresh test V2

## Registered correction

V1 selected rank eight because it retained `.999655` of aggregate discovery
response norm.  It transferred at `.02629` overall relative L2 and `.999654`
cosine with exact analytic and standalone checks, but failed the all-pair and
mixture composition gates.  Off-diagonal pairs carried only `.00366` of total
fresh response energy, so the aggregate energy rule discarded precisely the
small terms needed when diagonal responses cancel.  Rank 16 was already a
registered V1 candidate and retained `.9999911` on discovery; it was not opened
on a rank-16 fresh panel.

Freeze rank 16 without changing inputs, formula, discovery rows, eigensolver,
or package interface.  Evaluate on a second 64-prefix score-blind cross-domain
panel constructed after V1.  Keep the complete 4×4 response table and the same
eight fixed mixture coefficients as V1.  This is a prospective rank correction,
not a rescore of V1's opened panel.

## Predictions

- **A — analytic instrument:** uncompressed formula versus mixed JVP and swapped
  symmetry are each within `2e-5` overall, every family, and every pair.
- **B — registered rank:** rank 16 retains at least `.9999` discovery response
  norm.
- **C — second-panel prediction:** overall and every-family relative L2 are at
  most `.015`, cosine is at least `.999`, and every ordered pair is at most
  `.15` relative L2.
- **D — compositional reuse:** every fixed mixture is at most `.03` relative L2
  and at least `.999` cosine against direct mixed JVP.
- **E — correction value:** relative to the immutable V1 receipt, overall error
  and maximum pair error each fall by at least 50%; rank-16 error is at least
  `.20` below the median matched random-output control.
- **F — standalone extraction:** isolated CPU replay is at most `2e-6`, packaged
  tensor count remains at most `.70` of native MLP9 bilinear matrices, and the
  package retains exactly one native activation port.

Passing establishes a response-level node with cross-panel prediction,
standalone extraction, and exact bilinear reuse.  It does not establish that a
behavior reads the node or that removing it is selective.

## Price

One checkpoint load; the same 48 opened discovery prefixes; 64 second-panel
fresh prefixes; 16 ordered-pair and eight mixture mixed JVPs per fresh prefix;
one fixed rank-16 eigenspace export; 16 rank-matched random controls; one
isolated CPU replay; no suffix, logit, behavioral outcome, optimization,
parameter update, or removal intervention.
