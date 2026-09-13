# Shared interaction on a native strength grid

Before execution: fixed historical rows0(regional) and96(FineWeb), all16 multiplier pairs from{0,.5,1,1.5}². Multiply the frozen child/remainder scalar fields independently. One pristine native prefix and one prepared shared context per row; independently execute native MLP9/attention10 branches at each strength.

A: generated full-cross relative error<=0.001 on every nonzero pair; all pairs with either multiplier zero produce exactly zero generated cross. B: generated versus native full-cross local target effects have<=5% aggregate relative L2 error across the nine nonzero pairs for the regional row. C: no regional target sign reversals where the native effect magnitude>=1e-5. FineWeb all outcomes/signs/absolute and relative errors reported without a uniform tiny-effect guarantee.

For each pair, native additive postMLP10 background is h(child)+h(remainder)-h(pristine); compare background, background+nativecross, background+generatedcross through the native suffix. The candidate generates changed attention and joint RMS, but the evaluation background remains supplied. No new language OOD, corpus fitting, globally identified factors, or whole-circuit sufficiency claim. Two-thread CPU,120second limit. Preserve all32cells and original registrations if a bar fails.
