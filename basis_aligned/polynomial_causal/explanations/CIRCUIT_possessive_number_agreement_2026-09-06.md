# Circuit write-up: `possessive_number` — agreement, and what breaks it

**Date:** 2026-09-06 · **Model:** GPT-2 small · **Lane:** Claude (breadth)
**Receipts:** seven screens under `circuits/fast_screens/possessive_number_*_v1_result.json`
**Verdicts:** five `selective_causal_site`, two `native_behavior_incapable`

This is the most heavily controlled behaviour in the corpus: seven screens, each varying exactly
one property against a matched sibling, built specifically so that the failing configurations
could be diagnosed rather than merely recorded.

---

## 1. The behaviour

The number of an antecedent noun must be carried to a possessive pronoun.

| hypothesis | base | donor | answer |
|---|---|---|---|
| **A1** | `The clerks lost` | `The clerk lost` | ` their` / ` his` |
| **A2** | `In the notes the clerks listed` | `In the notes the clerk listed` | ` their` / ` his` |
| **P** | `The clerks checked` | `The members checked` | ` their` (both) |
| **C** | canonical same-answer control v2 | (different place/subject) | ` night` (both) |

Base and donor differ only in the antecedent's number and end on the same verb token.

---

## 2. The counterfactual series — one property varied per screen

All numbers are **interchange interventions** (patch the site from the donor run, measure movement
toward the donor answer). **No DAS has been run on this circuit yet** — see §4.

| configuration | distance | intervener | verdict | A1 | P | C |
|---|---|---|---|---|---|---|
| adjacent antecedent | 1 | none | **SELECTIVE** `resid:18` | 1.000 | 0.194 | 0.178 |
| medial antecedent | 4 | one PP (`at the desk`) | **SELECTIVE** `resid:18` | 1.000 | 0.158 | 0.183 |
| long simple intervener | 7 | two stacked PPs | **SELECTIVE** `resid:18` | 1.000 | 0.140 | 0.162 |
| inanimate argument | 5 | inanimate direct object | **SELECTIVE** `resid:16` | 1.004 | 0.200 | 0.195 |
| verb-final, distance six | 6 | VP + object, ends on a verb | **SELECTIVE** `resid:16` | 1.005 | 0.184 | 0.188 |
| **animate attractor** | 4 | **animate, number-mismatched** (`beside the manager`) | **NULL** | — | — | — |
| **particle-final** | 6 | VP + object, ends on a particle (`put away`) | **NULL** | — | — | — |

### What the series establishes

Two disruptors, each demonstrated against a matched control that differs in one property:

1. **An animate, number-mismatched intervener.** Fails at distance 4, where an *inanimate*
   mismatched nominal at the same distance passes. The failure is graded, not a collapse: A1 at
   57/64 rows with cells 0.81–0.88 against a 0.85 bar.
2. **A particle-final prediction site.** Fails at distance 6, where a *verb*-final site at the
   same distance and the same VP+object structure passes.

Three explanations were tested and **ruled out**, each against a matched control:

- **Token distance** — obliques pass at distance 7, longer than the failing distance-6 case.
- **Argument prominence** — an inanimate direct object passes at distance 5.
- **"Controller locality"**, my own original framing — retracted after the distance-7 result.

### Method note

Three of the seven screens fired the branch *against* the hypothesis held at the time, including
two that retracted claims posted within the previous hour (a "4–6 token threshold", and the
locality framing). Each screen registered **opposite branches in its prior-art receipt before
running**, so it could not confirm whichever story was current. That is why the series converged.

---

## 3. Circuit path and grain

At the selected site the whole variable is recoverable; no single block carries it.

| site family | best site | A1 recovery |
|---|---|---|
| residual stream | `resid:18` | **1.000** |
| attention block | `attn:08` | 0.322 |
| MLP block | `mlp:08` | 0.207 |

Passing band `resid:07`–`resid:18` in the adjacent design; the onset moves later as the intervener
grows (07 at distance 1, 10 at distance 4).

**Grain reached: whole residual-stream site.** No head-level or MLP-level follow-up has been run on
this behaviour, and the block-level maxima (0.32 / 0.21) mean no single block is a candidate
mechanism. Sub-block structure is unresolved.

---

## 4. DAS: queued, not done

The causal evidence above is interchange intervention only. Directed at the natural next question
— *which subspace at `resid:18` carries number?* — the corpus currently has no answer, because no
Distributed Alignment Search has been run on it.

**This is now a standing follow-up step** (see `ops/README.md`, "DAS follow-up on localized
circuits", and the board note of 2026-09-06). The target here is well posed: `resid:18` recovers
1.000 with a clean P (0.14–0.20) across five matched configurations, so there is a stable site to
search within, and the two established disruptors give ready-made negative controls — a subspace
that genuinely encodes number should survive the inanimate-intervener configurations and degrade
under the animate attractor.
