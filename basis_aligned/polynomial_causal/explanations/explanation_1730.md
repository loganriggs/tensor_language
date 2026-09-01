# Plain-English update — 2026-09-01 17:30 UTC

**Headline:** the first layer's investigation is finished — not abandoned, *finished* — with a rare kind of ending: a mathematical identity for exactly what compression loses, and proof that nothing cheaper than rank can buy it back.

**The last chapter.** After routing died five ways, the team found real hope on the output side: a small basis capturing 40% of the compressed layer's error, priced below the next rank step. Then came the beautiful part — the error turned out to be *exactly computable from the model's own weights* (a quadratic formula, no learning needed, matching the ideal correction almost perfectly). And then the sobering part: that formula, compressed to any affordable size, keeps too little; and when every candidate was priced against "just buy more rank instead" at *matched* cost, rank won at every single price point, by growing margins.

**Why this is a good ending.** Most investigations trail off; this one closed. The layer now has: a complete causal grammar (what it computes, for whom), an exact formula for the compression residual, six signed certificates listing everything that provably doesn't work, and one clean measured curve — the rank/damage tradeoff — that is provably the best available deal. Any future idea for this layer must beat that curve or cite those certificates.

**Meanwhile:** the compressed-model dial (four tiers, down to 495.8M parameters) stands untouched by all this; the day is at ~115 experiments with zero loosened thresholds; and the pipeline itself got smoother (rerun-causing bookkeeping now has a single canonical code path).
