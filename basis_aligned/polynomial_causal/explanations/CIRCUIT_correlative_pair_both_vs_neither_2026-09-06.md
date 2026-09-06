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

## 4. DAS result: a **single direction** carries the correlative state

Run: `circuits/followups/das_correlative_pair_resid18_rank1_v2_result.json`
Tool: `ops/circuit_das_subspace.py` · Runner: `ops/run_das_correlative_pair_rank1_v1.py`

**Rank was fixed at 1 before running** and registered with the prediction, per the protocol in
`ops/README.md`. It was not raised.

### Method

`resid:18` is the final residual site, so the map from it to the logits is only the model's own
head — `logits = 30 * tanh(lm_head(rms_norm(x)) / 30)`, verbatim from
`jacclust/tt_model.py:257-260`. That is exactly differentiable and needs no transformer forward
inside the optimization, which is what makes rank-1 DAS cheap here.

An orthonormal R (1152x1) is learned so that patching **only** the projection

    x_patched = x_base + R R^T (x_donor - x_base)

matches the donor's answer margin. Fit on 16 A1 rows; everything below is evaluated on rows the
fit never saw.

**Instrument control, run before any fitting:** the differentiable head reproduces the producer's
own native answer/foil values to **5.7e-06**.

### Results

| family | measure | value | whole-site reference |
|---|---|---|---|
| **A1 (held out)** | interchange recovery | **0.980** | 1.000 |
| **A2** | interchange recovery | **0.821** | 1.000 |
| **P** | same-answer effect | **0.053** | 0.024 |
| **C** | same-answer effect | **0.001** | 0.053 |

All four registered predictions hold: the head is the model's, one direction transfers on
held-out rows, it transfers **across constructions**, and it is selective.

### What this establishes

**The open correlative is carried by a one-dimensional subspace of `resid:18`.** A single learned
direction recovers 98% of what the whole 1152-dimensional site recovers, on rows the fit never
saw, and it still recovers 82% in the *other* construction — so it is a carrier of the variable,
not a direction fitted to one frame. Patching along it leaves the answer-preserving edit and the
unrelated control essentially untouched (0.053 and 0.001 against a 0.2 bar).

This is the first sub-site localization in the corpus: interchange said "somewhere in `resid:18`",
and DAS says "one direction in it".

### Honest limits

- **A2 at 0.821 is the real test, and it is not 1.0.** The direction was fit on A1 rows; ~18% of
  the cross-construction effect is not carried by it. Some construction-specific structure
  remains outside the direction.
- A1 held-out at 0.980 is a weaker test than A2, being the same construction as the fit.
- **Two earlier defects, both mine, both caught by running the instrument** (`..._rank1_v1_result.json`
  is retained on disk as the invalid record): the first objective *maximized* the donor margin
  rather than matching it, which overshot to recovery 2.208 — and is doubly wrong because the
  head is logit-soft-capped, so climbing toward the cap flattens the gradient. The first P/C
  measure divided by `(m_donor - m_base)`, which for a same-answer family is legitimately near
  zero, reporting P at 24.678. Both are fixed; the numbers above come from the corrected run.
## 5. What the direction actually encodes: `neither`, not correlative state

Three further runs settled what the rank-1 direction is, and the answer is narrower than the
first result suggested.

| run | question | result |
|---|---|---|
| `das_correlative_shared_subspace_v1` | does the both/neither direction carry either/or? | 0.377 / 0.457 — **inconclusive** by preregistered thresholds |
| `das_correlative_joint_fit_v1` | does a direction fitted on BOTH pairs serve both? | 0.975 / 0.963 — **yes** |
| `das_correlative_neither_axis_test_v1` | does it carry a pair with NO negative member? | **0.036 / 0.020 — no** |

The confound was structural and visible in the stimuli: `both`/`neither` and `either`/`neither`
**both put `neither` -> ` nor` on the donor side**. A rank-1 direction separating {both, either}
from {neither} places `both` and `either` on the same side of itself, so it should not
distinguish them — and it does not, at 0.036.

To test that, `correlative_pair.both_vs_either` was authored specifically to remove the negative
member (`both` -> ` and` against `either` -> ` or`). It screens `selective_causal_site` at
`resid:17` in its own right (A1 1.000, A2 1.001, P 0.031, C 0.066), so the near-zero transfer is
not a weak behaviour — it is a direction that genuinely does not carry this contrast.

**Established:** `resid:18` carries a one-dimensional feature encoding **whether `neither` is
open** — the negative correlative specifically. It transfers across two pairs and two
constructions, and it is selective (P 0.053, C 0.001).

**Refuted:** that this is a general correlative-state feature. It is not. The `both`/`either`
contrast is carried by something else, at a different selected site (`resid:17`).

That is a narrower claim than "correlative state", and a better one: it names a specific lexical
feature with a clean causal signature rather than a category the evidence never supported.
