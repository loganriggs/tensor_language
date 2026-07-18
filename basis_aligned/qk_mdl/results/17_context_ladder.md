# The context-order ladder: why token-static fails where it fails

Logan's question (2026-07-20): get more *contextual* circuits bottom-up — MDL layer by
layer, exploit co-occurrence (a token with its own attention-out), use the tensor-network
structure — or dig into the earliest failure of token-static description and say
principledly why heuristics and weight-based information can't help. The ladder
experiment answers the dig-in question.

## The ladder

Order 0 (current program): tables indexed by the current token. Order 1: bigram tables.
Tested on the four earliest streams (attn0, mlp0, attn1, mlp1) — the sequence-determined
objects behind the W=1 wall (+0.888) — in two forms:

| form | coverage | R² gain (range over streams) | ΔCE, W=1 | ΔCE, W=2 |
|---|---|---|---|---|
| unigram (order 0, ref) | 100% | — | +0.888 | +0.443 |
| raw bigram + backoff (106k pairs, cnt≥4) | 53% | +0.12…+0.22 | +0.878 | +0.435 |
| pairclass-factored (256×256 dense corrections) | 99.8% | +0.09…+0.15 | +0.877 | +0.435 |

## Three findings

**NG-1: the estimation-cost wall.** Going up one context order squares the index set:
at 3.2M tokens, raw bigram rows average single-digit samples and cover half of test
positions. Each further order is worse. Any n-gram program is data-starved almost
immediately — this is a structural property of the ladder, not an implementation issue.

**NG-2: the pair-shaped context is behaviorally cheap.** The TN-factored version
(Logan's instinct: factor the pair index through embedding classes — dense, fully
covered, well-estimated) retains most of the variance gain and STILL buys ~0.01 nats.
So the dissociation is real and now confound-free: local-pair context explains a large
share of the early streams' *variance* but almost none of what downstream computation
*consumes* from them. The L2-vs-behavioral dissociation, at the context-order level.

**NG-3 (the answer to "why does it fail there").** What the early streams carry that
matters downstream is precisely the NON-local part: match-relevant identity at
long-range positions (the induction chain), position-specific content the selection
circuits compare — structure that no n-gram order captures *by construction*, because
its index is "where my previous occurrence was," not "what the last k tokens were."
That is exactly the residue the windowed-D live window preserves; the window isn't a
placeholder for a better table, it's the correct treatment for a component whose
natural index set is dynamic. And this is also the principled version of why
weight-based information can't help (EH-4's ρ=0.025): every object that matters is a
data-measure-weighted contraction — the pattern tensor against the corpus distribution
— and weights carry no measure; refining the *conditioning on data* (this ladder) at
least targets the right object, whereas refining the reading of weights does not.

## What this means for contextual circuits

The route to contextual circuit atoms is not finer context tables — it is the named
live components we already have (H5 match events, H7's scalar, mlp16's gains) plus
their interaction with the token-static skeleton, as in the circuit cards
(results/cards/). The pairclass corrections are cheap in ΔCE but their 65k cells are
still *nameable* (class-pair → correction direction) and may be useful descriptively;
they are saved (`../ngram2_pairclass.pt`) for card use where a pair effect is visible.

Files: `../ngram_tables.py/.json/.pt`, `../ngram2_pairclass.py/.json/.pt`.
