# Plain-English update — 2026-09-01 05:30 UTC

**The one-sentence headline:** we found that the "uncompressible" layers were only uncompressible in the wrong coordinate system, and the new coordinate system just produced our best artifact candidate yet.

**The story.** Our compressed model replaces pieces of the original 546M-number network with smaller factored pieces, and every replacement must pass a battery of 62 "certificates" — behavioral checks that specific circuits still work (lower added-error is always better). This morning we tried shrinking the input side of every MLP layer using the standard method (keep the directions with the most weight energy). It worked for 13 of 18 layers and exploded at the last three — one layer got 100× worse than typical.

**The twist:** the explosion was an artifact of *how we measured importance*. Weight energy says "these directions are big"; what matters is "these directions are used." When we ranked directions by how the layer's inputs actually vary on real text (contextual covariance), the same three layers compressed beautifully — the useful directions had been sitting at the *bottom* of the energy ranking. Same rank, same cost, 100–300× less damage.

**The other law we nailed down:** whenever we combine two individually-good replacements, the total damage is about 1.3× the sum of the parts — we measured 1.30×, 1.32×, 1.34× on three different combinations. And we proved you can't refit your way out of it: retraining downstream pieces on the combined model's actual signals recovered exactly nothing. The cost of composing is real interaction, not a bookkeeping error.

**Where that leaves us:** applying the better measure at the first layer produced a candidate that beats our current best artifact on all three axes at once — smaller (535.6M vs 536.9M numbers), less damage (+.0083 vs +.0090 CE added), more certificates passing (52 vs 50). It's now in final testing (out-of-distribution text, then a causal-intervention check). If it passes, it's the first time a new artifact strictly dominates an adopted one.
