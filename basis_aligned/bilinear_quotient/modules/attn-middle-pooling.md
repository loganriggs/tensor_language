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
Per-head: trivially high (const ≈ full for most). Collective: first partial stand-in (§1099) —
a per-head static DISTANCE KERNEL (values dynamic) recovers 0.583 of the middle band's collective
value (gatherer L3-5 only 0.394 — more routing-adaptive). Two-term kernel+content-sim model is
the registered next rung. Split-half audit: all collective numbers stable (§1101).

## Gotchas
- NEVER evaluate this band per-head and conclude it's dispensable (5.7× collective factor).
- Head-mean pattern analyses dilute specialists (§1085 caveat).
- Banding vs output-ablation flip on redundant parts (§1008-1009).

## Open
- Which ensembles carry the content seed — CLOSED at read grain (§1222): no compact ensemble;
  best 12-head greedy set = 43% of the 0.176 prose budget; the only nameable edge is the
  copy/induction core (2.5/3.8/5.5 = 23%); the rest is ~150 heads at ≤0.002 each. The
  §1093/§1187 collective picture confirmed at ensemble grain; crowd_scaling (queued) prices
  the tail's redundancy curve.
- The depth-growing content-sim bias (§1085): mechanism unknown.

## Value-range map (§1186-1189, complete at layer grain)
Read-masking @W64 (pos-0 visible), nats: L0 0 / L1 .004 / L2 .018 (induction trigger — the
front's only distant reader) / L3 .006 / L4 ~0 / L5-8 ~.009 each (REDUNDANT crowd: singles sum
.037 vs band joint .067, no indispensable layer, L5 not special once pos-0 visible) / L9 .002 /
L10-14 band .034 / L15-17 band .013 (readout local, §1153 triple-confirmed). All-18 joint .176.
Family-general shape (swiglu18: mid1 top .075; fingerprint = deeper spread, mid2 .063).
Whole-model long-range budget: .082 @W128 (selection .014 + values ~.07), smooth to zero at
trained context. Every MLP is a ≤64-token window function (§1183-85). Do not chase finer
carrier decompositions of the pool — redundancy closes it (population-code law, range edition).
