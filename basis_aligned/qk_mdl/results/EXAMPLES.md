# Qualitative examples: what the reductions actually look like

Referenced from [EXPLAINER.md](EXPLAINER.md). Every example below is produced by a committed
script; file pointers at each section.

## 1. Token classes from a folded QK head (vq16 on L0H3, bilin18)

The shared $[\hat q|\hat k]$ partition at $k{=}16$ produces classes that read as word-shape /
morphology / register groups — selection genuinely runs on "kinds of tokens." Nearest-to-
centroid exemplars (full listing: [vq16_exemplars.txt](vq16_exemplars.txt)):

| class (size) | exemplars | reading |
|---|---|---|
| c5 (305) | `The, The, Our, our, the, your, his, my, their, whose` | determiners/possessives |
| c15 (1604) | `386, 613, 75, 154, 173, 189, 147, 1923` | bare numbers |
| c7 (3604) | `Alps, 1888, 614, ASA, Byrne, Oblivion, 1906` | proper-noun/number mix |
| c6 (5184) | `ngth, comings, otomy, alyses, orate, rency, ising` | word-tail fragments |
| c2 (3169) | `conqu, ufact, depl, priv, hect, princ, disemb` | word-head fragments |
| c1 (5667) | `practicable, exclaimed, overlooked, and, exited` | mid-sentence content words |

The same head's second branch partitions differently (e.g. a places/names class:
`Sharma, Siberia, Tasmania, Oslo, Trudeau, Sicily`) — branches specialize.

## 2. The induction conjunction, decoded (tiny model, conditioned metric)

The pre-registered conjunction test's payoff: with data-conditioned factors, the identity
conjunct becomes visible as literal query→key token matching. Query token → top-3 matched
previous-key tokens (`*` = exact self-match; hit rate 0.444 over covered vocab; full listing:
[conjunction_examples.txt](conjunction_examples.txt)):

```
' People' -> ' People*', ' data', 'vertise'
'ividual' -> 'ividual*', 'ray', '�'
' signed' -> ' signed*', ' coach', '�'
' loc'    -> ' loc*', ' considered', ' sat'
' 12'     -> ' 12*', ' requ', ' '
```

Generic (unconditioned) weight analysis finds this structure at chance level — the
correlational/causal gap that reappears at every scale (see §4).

## 3. bilin18's two contextual heads (results/12)

**L5.H5 — the match head.** Induction signature = mean pattern on "key follows my previous
occurrence" positions over the unconditional mean: **16.8×** on natural text, **53×** on
repeated sequences; positional profile nearly flat to Δ64. Its carried content decodes to the
source token (logit-lens median rank **25**/50k). But it's carried *noisily*: cleaning the
content to cond-means improves repeat CE by −0.17, and low-rank filtering by −0.33 — while
amplifying the head (×4) costs +3.37. The model under-weights its own induction signal
because the signal is noisy.

**L5.H7 — the gain head.** No match signature in any context; local profile (high through
Δ≈4). Zero-ablating it: +1.04 natural / **+6.68** on repeats — the causal heavy lifter. Yet
its per-token mean content decodes to the same generic connectives for *every* source token:

```
' Al'     -> '-', ' (', ' and', ',', ' in'
' cliffs' -> ' and', ' so', ' all', '-', ' ('
' command'-> ' (', '-', ' in', ' all', ' and'
```

Its deviations are ~5% of output energy, 63% in ONE direction — and rank-1 replacement with a
live scalar costs **+0.0001**. H7 = one fixed hub-feature direction × one context-computed
number. Files: `../l5_heads_function*.json`, `../l5_h5_causal.json`, `../h7_ov_probe.json`,
`../h7_rank1.json`.

## 4. mlp16's contextual gains — and the trap they set (results/13)

Top deviation directions of the dominant top-MLP, with extreme-firing contexts:

| dir (var) | fires on | example context |
|---|---|---|
| 0 (40%) | legal-citation continuations | `…609, 614 (1965); see also` |
| 2 (8%) | legal captions | `…Plaintiff-Appellee\n\nDouglas K.` |
| 3 (5%) | XML/Android markup | `…EndOf="parent"\n    app:` |
| 7 (0.5%) | patent/numeric | `….S. Pat. Nos. 4,205,` |

These *look* like slow document-register state — but the causal swap test refuted that:
document-mean coefficients cost +0.103 (≈ deletion, +0.113) vs +0.023 live. The directions
fire in register-specific contexts while carrying **fast token-to-token structural position**
within those registers. A worked example of why lens+examples alone mislead: the correlational
identity of a direction and its causal content are different questions. Files:
`../mlp16_dirs.json`, `../mlp16_register_swap.json`.

## 5. Attention patterns from a compressed model

[fig_pattern_display.png](fig_pattern_display.png) shows live vs vq-classed layer-0 patterns
side by side on real text with class annotations (from the pre-redirect program; file 08).
The class-level pattern is visibly the same selection structure — the compressed model isn't
approximating the pattern pointwise, it's computing the same class-interaction rule.

## 6. The wall arc in one figure

[fig_wall.png](fig_wall.png) (routes to token-static selection + why re-estimation fails) and
[fig_window_ladder.png](fig_window_ladder.png) (every W-ladder, both models, trained walls as
reference lines — the untrained windowed curves crossing below the trained walls is the
flagship result in one image).
