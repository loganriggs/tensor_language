# Middle attention (L3-14) — redundant collective pooling; positional-first criterion

**One line:** individually near-worthless heads (max const-cost ~0.02) whose COLLECTIVE dynamics
are worth ~3 nats — a recency-first, weakly content-biased pool that feeds the content machine.

## Established facts
- **Per-head: thin.** Zero costs mostly <0.01 (§1083); const-replacement (true dynamics) max
  ~0.02 in the middle (§1091). Most middle heads classify as inert/pooler.
- **Collectively: load-bearing.** All-162-heads-static costs 3.67 nats = **5.7×** the per-head
  const sum (0.645); restoring the MIDDLE band's dynamics recovers 0.465 of it > front 0.374 >
  the 6 named routers 0.357 (§1093). Same super-additive theme as §931/§952-954/§1049.
  **Stale biases are worse than nothing:** all-zero 3.42 < all-const 3.67 (§1093).
- **Criterion (§1085 + §981-984):** middle |pattern| is POSITIONAL/recency-first (log-dist r
  −0.39..−0.61 top at every layer tested 6/8/10/12); content-similarity is second-order but real
  and GROWS with depth (r 0.08→0.18, L8→L12; masking top-content-sim keys costs 1.8-4.1× random).
  §983 (long-range control): apparent content-similarity routing was largely a recency confound.
  QK = signed conjunction of two bilinear scores; anti-heads exist (§981-982).
- **Function:** the pool is how the content bag is gathered — topic emerges from broad
  recency-weighted pooling of value-residual content (§995-998, §1074, §1076), read out locally
  late (§997). Middle attention is NOT the author of the content-read subspace (§1053); window
  stand-ins: middle attn is local routing of the residual, non-local remainder low-stakes
  (§1054/§1069).

## Benchmark status
Per-head: trivially high (const ≈ full for most). Collective: the honest open number — no
stand-in reproduces the 3.67-nat collective function yet (window/pool stand-ins recover the
front+gatherer part 0.66-0.95, middle remainder low-stakes §1054).

## Gotchas
- NEVER evaluate this band per-head and conclude it's dispensable (5.7× collective factor).
- Head-mean pattern analyses dilute specialists (§1085 caveat).
- Banding vs output-ablation flip on redundant parts (§1008-1009).

## Open
- Which ensembles carry the content seed (§1074 says early attention; L5H7 retired §1089) —
  needs ensemble-level intervention, not per-head (FINDINGS Open A).
- The depth-growing content-sim bias (§1085): mechanism unknown.
