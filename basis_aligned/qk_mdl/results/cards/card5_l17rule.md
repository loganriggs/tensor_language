# Circuit card 5: the L17 pronoun -> temporal rule

**The first cross-class rulebook entry, causally verified.**

Rule block (freq-filtered top off-diagonal at L17):
- queries: [' everything', ' anyone', ' they', ' what', ' everyone', ' What']
- keys: [' when', ' after', ' while', ' After', ' until', 'While']

## Causal probe (block-ablate at L17, all heads; audit at pronoun-query positions, n=624)

| arm | Δlogit on temporal-conj class | Δlogit other tokens |
|---|---|---|
| RULE block ablated | **+0.0016** | +0.0009 |
| control block 0 | +0.0001 | +0.0001 |
| control block 1 | -0.0000 | -0.0001 |
| control block 2 | -0.0003 | -0.0006 |

## Verdict

**Real, selective, featherweight — and suppressive.** The block's ablation effect is
~5× matched controls (+0.0016 vs ±0.0003) and ~2× selective for the temporal class
over other tokens — the rulebook entry is causally REAL. But it is milli-logit scale,
and its sign shows the live block mildly SUPPRESSES temporal conjunctions at pronoun
positions rather than promoting them. Reading: top-layer rulebook entries are
individually featherweight routing adjustments — the rulebook is meaningful as
aggregate structure (97% of it deletes for free, results/19), not as a set of
single-block behavioral levers (consistent with CP-1, results/18). The map is real;
its atoms are small.