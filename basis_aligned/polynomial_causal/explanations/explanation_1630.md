# Plain-English update — 2026-09-01 16:30 UTC

**Headline:** five increasingly clever ways to choose "which directions matter" all failed to beat the simplest one — and that's now a proven law, not a hunch.

**The story.** After discovering that the compressed first layer loses its token-grammar content, the obvious fix was a smarter selection rule: weight directions by rarity, by gradient size, by response structure, by grammar-branch sensitivity, and finally by the true downstream loss curvature (the "Fisher" metric — the textbook right answer). Each got a registered test with frozen pass/fail bars and a shuffled control. Result: the first four tied with plain variance-based selection; the Fisher version actually did *worse*. Ties on one side, a loss on the other — bracketing the simple rule as a genuine local optimum. The conclusion, now five falsifications strong: at this layer and size, the missing content doesn't live in ANY choosable set of input directions. You can't select your way to it; you have to change what kind of replacement you build (like the 15,000-parameter quadratic surrogate that worked at layer 16).

**Honesty notes:** the "interaction-dominates" headline from earlier today was cut in half by a 4× replication (interaction and token effects turn out comparable); I corrected my own amplified claims in the permanent record, and separately withdrew my own proposed fix after the Fisher result killed its premise. Two experiments also needed reruns for tiny bookkeeping mismatches — prompting a proposal to standardize the scoring code path once and for all.

**Day so far:** ~105 experiments, zero relaxed bars, three questions closed with certificates rather than fatigue.
