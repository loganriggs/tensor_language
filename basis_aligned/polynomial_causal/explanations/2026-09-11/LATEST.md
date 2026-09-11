# Latest research update

**Current completion: both penalized starts finished at 13:29:35; capture 64.6851/64.6855%, convergence missed, function cosine 0.74484. Details in the requested report below.**

**Latest requested full report: [11 September, 13:27 UTC](../for_logan/research_update_2026-09-11_1327.md).** Requested reports now live in [for_logan](../for_logan/README.md); the dated notes below are historical or automatic updates.

**11 September,13:10 UTC:** [Weights-first dictionaries perform better on cached text despite a worse coefficient score](explanation_2026-09-11_1310.md).

A native-product baseline uses slightly fewer parameters and captures68.63%ofcoefficients versus65.00%for the dictionaries. But on128fixed cached FineWebpositions, its next-token cross-entropy damage is**+0.355nats**, versus**+0.012/+0.036**for the two dictionaries; KL isalso muchlowerfor the dictionaries. Allprograms were frozen before validation. This supports a functional advantage on this small historical panel, not fresh/OOD confirmation or identified circuits.

First jointpenalized start:64.685%capture andfivefold less summed component energy, but still unconverged. Secondstart is running. [Receipt](../../PENALIZED_PROJECTED_FIT_V1_SEED_0.json).

The shared scalar candidate substantially overlaps knownpronoun-related structure and has a strong opening-parenthesis association. It isnot a newgendercircuit. [Prior comparison](../../SHARED_FUNCTION_PRIOR_ALIAS_V1_AUDIT.json).

[Methods explanation12:22](explanation_2026-09-11_1222.md) · [Hourly12:22](../../HOURLY_STRATEGIC_REVIEW_2026-09-11_1222.md) · [Math10:51](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_1051.md) · [Method index](../../WEIGHT_ONLY_METHODS_INDEX.md).
