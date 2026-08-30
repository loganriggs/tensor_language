# Induction and repeated-bigram copying

## CURRENT tier: 4

The matcher, identity payload, relays, and product algebra are known; not every required
source has been recursively replayed to embeddings in one terminal program.

## Behavior and tensor program

Endpoints are synthetic `AB ... A -> B` change in `log p(B)`/CE and natural CE where
the current/next bigram occurred earlier.  Matched negatives repeat `A` with a different
earlier follower; first mentions and noninductable repeats are controls.

Fixed tensor form: double-bilinear QK matchers and rank-128 identity OV payloads in
`L5H5/L7H3/L8H3/L8H4`, with L4H7/L6H3 relays.  The **router is the source-match QK
score itself**.  Extraction executes native/reconstructed scores; selective removal
gates only the matched source edge.  An unconditional mean router is forbidden.

## Evidence

- [`induction_mechanism.py`](../../induction_mechanism.py),
  [`circuit_induction.py`](../../circuit_induction.py), and
  [`induction_injection_family.py`](../../induction_injection_family.py).
- [`terminal_copy_selection_v1_attempt2_result.json`](../../../polynomial_causal/terminal_copy_selection_v1_attempt2_result.json):
  four-head target effect `+0.44869993` nat but collateral margin `-0.01440914`, rejecting
  unconditional mean replacement.

## Terminal gates

Extraction retains/rebuilds the four heads and relays in an attention-null background.
Removal is source-match gated.  Collateral includes first mentions, wrong followers,
copied non-targets, entity/capitalization, successor, and global CE.  OOD holds out
token identities, lag bands, natural/synthetic roles, and domain; default gates apply.

Shared-owner caveat: the copy heads overlap copied-entity and capitalization service.

**Next experiment:** recursively replay matcher/payload/relay sources to token and
position primitives, then run source-match-gated terminal extraction/removal.
