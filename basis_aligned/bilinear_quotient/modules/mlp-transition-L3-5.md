# Transition MLPs (L3-5) — content onset; L4 the first true context MLP

**One line:** the band where content is born (§1052); heterogeneous — L3 still ~front-like,
L4 the first MLP that genuinely needs context, and mlp4 separately manufactures the sink constant
at position 0.

## Established facts
- **Content onset is smooth (§1052):** shared content directions rise L1-2 (0.16 overlap w/ deep
  ref) → L3-5 (0.26→0.44) → L6-12 (0.53→0.91). Neither token-only (§1045) nor bag (§1048)
  stand-ins rescue the transition MLPs — they catch the birth of the content flow.
- **NO binding band (§1084):** cross-terms (tok×dev) are flat ~0.2 variance share everywhere,
  never dominant — the transition does not specially multiply token against context; the depth
  story is a smooth tok→dev gradient (tok share 0.73 front → 0.34 deep).
- **L3:** still substantially token-driven (tok-only CE recovery 0.67 held-out, §1088).
- **L4 — RESOLVED (§1084/§1088/§1094/§1095):** tok-only recovery ~0.05; highest dev variance
  share of any layer (0.47). §1094: the DEEP (L8-12) content projection is neither necessary
  (strip it → 0.914 intact) nor sufficient (0.363). §1095: L4 consumes the content PRECURSOR —
  the same content variable in its LOCAL rotated coordinates: own-top-64 dev recovers 0.74
  (own-256: 0.93); own basis overlaps the precursor (L3+L5 dev) 0.645 = 2.0× deep ref, 3.7×
  grammar, 11× random; causally precursor-64 projection recovers 0.659 vs deep 0.363 / grammar
  0.229 / random 0.162. The §1052 drift is FUNCTIONAL: each layer reads the content object in
  the coordinates it has at that depth; content×content effectively begins at L4 in precursor
  coordinates. rare/freq of L4 mean-abl = 1.63.
- **mlp4's second job (§439 + sink arc):** at position 0 it writes the fixed constant (norm 155k,
  cross-doc cos 0.998) that head 5.7 broadcasts (see `attn-sink-5-7.md`). This is position-
  triggered, not token-triggered, and is invisible to per-token-mean analyses (pos 0 = 1/255 of
  positions, negligible in aggregate CE of ordinary-position experiments).
- **L5 MLP:** intermediate (tok-only 0.48, tok+cross 0.61 in-sample §1084). Per-layer stakes in
  this band are small individually (mean-abl L4 0.103, L5 0.084) — collective via the band.

## Benchmark status
L3 ~0.7 (token table, held-out); L4 the WORST-understood MLP relative to its own stakes (no
partial stand-in >0.36 except stripping content 0.91 — which identifies what it does NOT use);
L5 ~0.6.

## Gotchas
- Position 0 is special at mlp4 — exclude/handle it explicitly.
- The L8-12 content basis drifts (§1052): "non-content" at L4 may contain rotated early content —
  test in L4's OWN basis before concluding a different variable (registered §1094).

## Open
- L4's context variable identity (l4_variable.py: own-basis rank + precursor/grammar/deep overlap
  + causal projection tests).
