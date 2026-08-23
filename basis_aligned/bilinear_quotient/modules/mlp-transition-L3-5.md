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
- **L4 — the step (§1084/§1088/§1094):** tok-only recovery ~0.05 (nothing partial works;
  tok+cross 0.22); highest dev variance share of any layer (0.47). Its context variable is NOT
  the deep content manifold: removing the L8-12 topic projection from its input leaves it ~intact
  (recovery 0.914), while content-projected dev gives only 0.363 (3× random-64's 0.162 but
  insufficient) (§1094). rare/freq of its mean-abl = 1.63 (between machines). Identity of its
  variable = OPEN (precursor-content in rotated early coordinates vs a different local/class
  variable — l4_variable.py queued/§1095?).
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
