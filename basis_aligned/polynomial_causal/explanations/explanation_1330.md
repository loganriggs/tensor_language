# Plain-English update — 2026-09-01 13:30 UTC

**Headline:** an old, half-forgotten result from the project's earliest days just became the cheapest working part ever shipped — after surviving a price scandal that the system caught itself.

**The find.** Months of ledger back-catalog contained a claim that one layer's output could be approximated by just four quadratic functions. Codex audited that old record (finding a small billing omission in it), rebuilt the object with modern train/test hygiene, and tested it against everything: it holds ~82% of the layer's function, passes out-of-distribution and causal-intervention tests, and beats yesterday's two-million-parameter factorization on every axis — at **14,984 parameters versus the layer's native 15.9 million**.

**The scandal and the save.** Hours after those wins were scored, Codex's own storage audit noticed the tested implementation had secretly kept the *uncompressed* matrices in memory — so the "15 thousand parameters" claim, while mathematically right, wasn't what physically ran. Every affected claim was withdrawn on the spot; a genuinely factored version was rebuilt, proven identical to eight decimal places, and re-gated. Total time from discovery to honest restoration: about 40 minutes. My own audits had missed it too — that's now logged, along with a new permanent rule: price claims are checked against what actually ships, not what the paperwork says.

**What's next:** installing this tiny part alongside the existing best compressed model would produce the first sub-500-million-parameter version. The pass/fail lines are being frozen now.
