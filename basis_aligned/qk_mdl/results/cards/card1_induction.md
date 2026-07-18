# Circuit card 1: induction copy of a repeated rare name

**Cherry-picked example, with set-ablation verification** (guardrails per LOG tick 68).

## A. The behavior

Prompt: `The merchant Dunleavy counted his coins twice before speaking. Nobody in the market trusted Dun`

Prediction position: the second ` Dun` (pos 17); target `le` (the continuation seen after the first occurrence).
Baseline logP(target) = **-0.005** (rank 0 of 50k).

## C. Live components at the prediction position

- **L5.H5 (match head)** attends hardest to positions [3, 4, 5] = ["'le'", "'avy'", "' counted'"] — the post-first-occurrence position is 3 (HIT).
- H5's head-output logit-lens at this position decodes to: ["'ch'", "'naires'", "'k'", "'landers'", "' gau'"] (target is `le`).
- **L5.H7 (gain head)** attends locally to [0, 16, 12] (["'The'", "' trusted'", "' Nobody'"]); its output is the usual structure-gain direction (rank-1, results/12).

## D. Causal set-checks (this prompt)

| ablation | ΔlogP(target) |
|---|---|
| {L5.H5, L5.H7} together | **-3.383** |
| L5.H5 alone | -0.002 |
| L5.H7 alone | -6.386 |
| {L5.H0, L5.H3} (matched random) | +0.001 |
| {L9.H2, L12.H6} (random elsewhere) | -0.000 |

## B. Static skeleton (what the tables say about `Dun`)

- `emb` table row for `Dun`: nearest peers ["'Dun'", "' dun'", "' Duncan'", "' Hun'", "' Dunham'", "' Hor'"]
- `mlp0` table row for `Dun`: nearest peers ["'Dun'", "' Dul'", "' Bon'", "' Mont'", "' Shar'", "' Har'"]
- `attn5` table row for `Dun`: nearest peers ["' Bur'", "' L'", "' Ch'", "' Mel'", "' Ald'", "' Bos'"]
- `mlp4` table row for `Dun`: nearest peers ["' Wor'", "' Ald'", "' Bur'", "' Sal'", "' Ly'", "' Hel'"]

Load-bearing edges these rows ride (from the edge map, results/15): emb→L1 reads, mlp0→L1, mlp4→L5 (+0.62), attn5→L5 (+2.61) — the short-hop chain that delivers `Dun`'s identity into layer 5, where H5 does the match and H7 the transport.

## Verdict

The format works, and the numbers are honest rather than tidy:

1. **Selectivity confirmed**: the traced pair matters (−3.38); matched random head pairs do nothing (±0.001).
2. **H7 dominates causally on this prompt** (−6.39 alone; H5 alone −0.002) — exactly the WW-2 pattern: the match head's signal is real (it attends to the right position — the `le`/`avy` continuation of the first occurrence) but under-cashed; the transport head carries the behavior.
3. **Even two heads interact non-additively**: ablating BOTH (−3.38) is *less* damaging than ablating H7 alone (−6.39) — with H7 gone, removing H5's (noisy) contribution partially helps. The composition law reaches all the way down to a two-element circuit.
4. **The skeleton reads well**: `Dun`'s embedding peers are a name-prefix class (`Dun`, ` Duncan`, ` Dunham`); by attn5/mlp4 the peers blur to generic name-prefixes (` Bur`, ` Ald`, ` Mel`) — identity dissolving into class as it rises, which is "selection needs classes, carriage needs identity" made visible in one token's journey.

Caveats: single cherry-picked prompt; H5's output logit-lens decodes junk (crude lens + noisy identity payload, as expected from results/12); this model's induction is weak overall. Next cards should include a non-induction behavior and a prompt where H5 is load-bearing (repeat data).