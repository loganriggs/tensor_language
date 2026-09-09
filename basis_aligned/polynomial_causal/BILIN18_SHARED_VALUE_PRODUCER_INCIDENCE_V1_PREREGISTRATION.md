# Exact mixed-command test for any token-only value producer

Use all32 existing dual-command-v2 worlds in their unchanged four-cell order.
At each aligned source position form the integer token-incidence contrast
delta=count(token00)+count(token11)-count(token01)-count(token10).
If delta is identically zero, every token-only map phi has zero mixed
command contrast at that source. This includes native v0 by the inspected
architecture: its initial normalizations, scalar mixture and V0 projection
are token-local and occur before attention. No numerical weights are needed.

A: equal four-cell lengths, fixed builder hash,32worlds, exact positive
fixture with independent token positions, and a live XOR-token negative.
B: all source-position contrasts are zero. If B fails, preserve every
nonzero position and do not claim a separable producer.

Combine B only with already-opened shared-response mode evidence. A nonzero
mixed output-removal mode with a zero producer mode places the interaction
downstream of the token-only producer; it does not identify one head or prove
the native attention read alone explains the final response.

Also report the already-opened M11-only lower bound for any additive response
F(a)+G(b), from orthogonal projection onto modes00/10/01. This is a diagnostic
function-class bound, not an adopted expansion of the failed invariant model.
No fits, source/head selection, native forwards or GPU. CPU bound30seconds.
All native parameters remain priced; no structural saving.
