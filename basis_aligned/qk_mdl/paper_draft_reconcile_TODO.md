# Paper-draft reconciliation items (from consolidation agent, 2026-07-30) — resolve before final
1. Programmatic-head count: 23/162 (census v1, 8 predicates) vs 30/162 (census v2, 12 predicates). Both
   correct for their version — use v2 (30) as current, footnote v1. NOT an error.
2. "programmatic heads" (census, >=5% predicate gain) vs "gated-nameable selection heads" (§49
   simultaneous-substitution gate) are DIFFERENT counts/criteria -> add a clarifying footnote so the
   driver-table head lists and the §49 per-layer counts don't read as contradictory.
3. Substitutability loss-layer L17 (per-minibatch positional-mean floor, §48) vs L5 (§12q full-corpus
   mean). Methodology difference; reconcile with a full-corpus mean rerun OR footnote both.
4. Diffuse-layer set differs by ledger: attention-function diffuse = 4/9/17 (no programmatic head);
   selection-GATE diffuse = 4/17 only (L9 has 1 gated head). Add one-line note (different criteria).
5. Whole-model substitutability: +0.0475 (PCA/head bottleneck) vs +0.0329 (MLP-chain-only) are DISTINCT
   experiments; per-layer marginal 99.95-99.998% vs cumulative whole-model ~98-99.8%. Keep separate,
   confirm the intended headline framing (per-layer marginal is the sweep result).
