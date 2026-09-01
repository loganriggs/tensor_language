# Plain-English update — 2026-09-01 08:30 UTC

**Headline:** the compressed model is now smaller than the original in BOTH currencies — half the storage bytes and 34 million fewer parameters — and the last improvement was *designed on paper first*, then confirmed in one shot.

**The design-first moment.** For weeks, finding a better compressed model meant running experiments and seeing what survived. This morning the two calibration laws (how much error a cut of a given size causes; how many behavioral certificates survive a given error) got good enough to invert: Codex computed which cuts to make *before* touching the GPU, froze the selection rule, and the physical build landed inside every predicted bar. Four steps, thirty-five minutes: discover all 18 MLP layers compress under the right metric → pick the two safest by a preregistered formula → add free precision reduction → pass the causal-faithfulness gate.

**A satisfying twist:** the new artifact isn't just cheaper — it's *more* causally faithful than the point it replaced (intervention-agreement .9865 vs .9816). Spreading compression thinly across several places beats cutting deeply in one place, even for faithfulness. That matched the water-filling theory's prediction.

**Also this hour:** storing everything in 16-bit floats was proven completely free (the full-precision and half-precision models agree to the 4th decimal even under interventions), so the "half the bytes" part costs nothing. And one clean negative: physically sharing an encoder between two layers works but wastes error budget — the layers genuinely use ~32% different subspaces, so sharing is the wrong economy here.

**Where it stands:** original model 2.07GB; current best faithful artifact 1.02GB with 43 of 62 certificates intact and every claim behind a pre-frozen test. The zero-damage version (identical behavior, all 62 certificates) is 1.09GB.
