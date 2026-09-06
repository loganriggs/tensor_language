# Circuit write-up: `correlative_pair.both_vs_neither`

**Date:** 2026-09-06 · **Model:** GPT-2 small · **Lane:** Claude (breadth)
**Receipt:** `circuits/fast_screens/correlative_pair_both_vs_neither_v1_result.json`
**Verdict:** `selective_causal_site` — the cleanest selectivity profile in the corpus

---

## 1. The behaviour

A correlative's first element commits the sentence to a specific second element. `both` obliges
` and`; `neither` obliges ` nor`. The model must carry which correlative is open across an
intervening noun phrase.

This is the corpus's second correlative pair. `correlative_state.either_vs_neither` established
that `either` → `or` is carried; this pair differs in kind, not just in tokens: `either`/`neither`
both open a two-way choice, whereas `both`/`neither` is a **polarity** contrast over a
conjunction. The answer vocabulary is disjoint from the earlier pair's.

### Stimuli (32 panels, 128 rows)

| hypothesis | base | donor | answer |
|---|---|---|---|
| **A1** bare frame | `The leader praised both the guide` | `The leader praised neither the guide` | ` and` / ` nor` |
| **A2** report frame | `In the notes the leader named both the judge` | `In the notes the leader named neither the judge` | ` and` / ` nor` |
| **P** invariance | `The leader praised both the guide` | `The member praised both the guide` | ` and` (both) |
| **C** control | canonical same-answer control v2 | | ` night` (both) |

Base and donor differ only in the correlative word and end on the same object noun. A2 uses a
different matrix verb, so a single lexical association cannot satisfy both constructions. The
object table is rotated off the agent table so the agent and object are never the same noun — an
early draft produced `The leader praised both the leader`, caught by printing one row per family
before running.

---

## 2. Counterfactual (interchange) results

All numbers are interchange interventions, as in the other write-ups. **No DAS run yet** — §4.

At the selected site `resid:18`:

| quantity | value | bar |
|---|---|---|
| A1 recovery | **1.000** | ≥ 0.5 |
| A2 recovery | **1.000** | ≥ 0.5 |
| direction fraction | 1.00 | ≥ 0.8 |
| **P invariance** | **0.024** | ≤ 0.2 |
| **C control** | **0.053** | ≤ 0.35 |

**P 0.024 and C 0.053 are the lowest of both in the canonical-v2 comparable set.** Ten passing
sites, `resid:09` through `resid:18`. For anyone choosing a behaviour with maximum headroom on
both the invariance and control clauses, this is the one.

### Circuit path and grain

| site family | best site | A1 recovery |
|---|---|---|
| residual stream | `resid:18` | **1.000** |
| attention block | `attn:08` | 0.276 |
| MLP block | `mlp:08` | 0.139 |

The open correlative is present in the residual stream from **layer 9** and fully recoverable by
layer 18. As with every behaviour in this corpus, **no single attention or MLP block carries it** —
the best attention block reaches 28%, the best MLP 14%.

**Grain reached: whole residual-stream site.** No sub-block follow-up has been run.

---

## 3. Class generality

With this screen the correlative class is **2 for 2 across pairs**, and verb subcategorization is
9 for 9 across four axes (clause type, preposition, finiteness, dative). Function-word and
subcategorization cues pass consistently in this model; the corpus's nulls cluster instead on
lexical-semantic cues (pronoun gender at chance, countability, animacy).

---

## 4. DAS: the natural next step, now queued

This behaviour is the best-conditioned DAS target in the corpus, for a specific reason: the
invariance and control clauses both sit far from their bars (0.024 and 0.053), so a subspace
search at `resid:18` starts from an unusually clean signal — there is little competing structure
at the site to confuse an alignment.

The concrete question DAS answers here and interchange cannot: **is the open correlative carried
in a low-dimensional subspace of `resid:18`, and is it the same subspace that carries
`either`/`or`?** If the two pairs share a subspace, the corpus has a genuine "correlative state"
feature rather than two separately-learned lexical associations — and that is a claim about the
model's representation, not just about which site to patch.

Registered as a standing follow-up step for all localized circuits (see `ops/README.md`, "DAS
follow-up on localized circuits").
