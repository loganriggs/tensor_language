# Front attention (L0-2) — the genuine routers, and a literal lookup table

**One line:** the front holds nearly all of attention's TRUE dynamic value (per-head), led by
L0H3 — and layer 0's attention is exactly a bigram table.

## Established facts
- **Layer 0 = a lookup table (prior sink arc, report §"first layer is a table"):** replacing
  head 0.3's pattern with one computed from tokens+positions alone, and its values with a
  50,304-entry per-token table, costs **zero** (shuffled table +0.15). All nine L0 heads at once:
  also zero (shuffled +0.24). The identical construction at layer 1 costs **+1.47** — the
  table boundary is sharp at L0/L1.
- **Dynamic ranking (§1091, const-replacement cost = true dynamic value):** L0H3 0.079
  (bias-frac only 0.10 — its value IS the routing), L2H5 0.028, L1H1 0.024 (61% bias),
  L6H3 0.023, L9H7 0.020, L7H8 0.018; nothing else >0.015. Front L0-2 = 35% of all dynamic value.
- **Zero-ablation costs (§1083 ≈ §429):** L0H3 0.088, L1H1 0.062, L2H5 0.036. Profiles: L0H3
  self/prev/local = 0.23/0.30/0.24 (prev-token/local router); L1H1 local; L2H5 is already a broad
  POOLER at L2 (consistent with early content onset §1052).
- **Window sizes (prior sink arc):** front attn + mlp0 together = a **bigram function** (current
  + one previous token costs 0.004; 4-token window free). 17/18 layers need <0.1 nats beyond a
  4-token window (exception = L5's position-0 constant fetch).
- **Roles vs bands (§1043-1047/§1054):** front attn = local-window routing OF THE RESIDUAL
  (window stand-ins recover 0.66-0.95); attn0 also the induction chain's copy-source writer
  (prev-token; §877/952).

## Benchmark status
L0 attention ≈ **100%** (token+position pattern + per-token value table, prior arc). L0H3
alone likewise. L1-2 routers: window stand-ins ~0.7-0.95 (§1054). Files:
`head_const_map_results.json`, `attn_head_map_results.json`.

## Gotchas
- L1H1 is 61% bias — don't treat the whole front as "dynamic".
- Per-head zero costs understate collective front function (§952: front-6 attention mean-ablate
  = +5.2 on inductable positions; §1093 collective factor 5.7×).

## Open
- L0H3's exact routing criterion beyond prev/local (likely just the bigram table fact above —
  check prior arc before building anything).
