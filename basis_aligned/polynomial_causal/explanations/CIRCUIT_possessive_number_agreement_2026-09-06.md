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

## 4. DAS result: one direction carries number across every intervening structure

Run: `circuits/followups/das_possessive_number_resid18_rank1_v1_result.json`
**Rank fixed at 1 before running** and registered with the prediction; not raised.

A single direction was learned at `resid:18` on the **adjacent** design only (distance 1, nothing
intervening), then evaluated untouched on the four matched siblings — which differ in what sits
between antecedent and pronoun, and in distance from 1 to 7.

| evaluated on | intervening material | recovery |
|---|---|---|
| adjacent, held out | none | **0.886** |
| adjacent, A2 construction | none | 0.837 |
| medial | one prepositional phrase (distance 4) | 0.832 |
| long simple | two stacked PPs (distance 7) | 0.735 |
| inanimate argument | an inanimate direct object (distance 5) | 0.704 |
| verb-final | a VP with its own object (distance 6) | **0.694** |
| P (invariance) | — | same-answer effect **0.147** |
| C (control) | — | same-answer effect **0.003** |

Registered thresholds: ≥ 0.50 minimum across siblings means a number feature; ≤ 0.15 means a
design-specific direction; between is inconclusive. **The minimum is 0.694.**

**Reading: `number_feature_survives_intervening_material`.** All registered predictions resolve
that way, and the instrument control — the differentiable head reproducing the producer's own
native values — passed at 2.9e-06 before any fitting.

### What this establishes, and what it does not

**Establishes.** `resid:18` carries a one-dimensional feature encoding the antecedent's number. It
was fitted on a three-token frame and still recovers ~70% of the effect across a seven-token gap
containing two prepositional phrases, and across a VP with its own object — structures the fit
never saw. It is selective: the answer-preserving edit reads 0.147 and the unrelated control
0.003.

**Does not establish.** That the direction is complete. Transfer declines with intervening
complexity — 0.886 adjacent, 0.832 one PP, 0.735 two PPs, 0.704 and 0.694 with a verbal object —
so roughly 20-30% of the effect in the harder frames sits outside it.

### The negative controls were deliberately excluded, and why

The two configurations that **fail** natively — the animate number-mismatched attractor and the
particle-final site — were left out on purpose. Their base-to-donor separation is unreliable
precisely because the model does not perform them, so a recovery ratio on those rows would be
noise presented as evidence. Whether the direction *degrades* under the animate attractor is a
real question that needs a measure not dividing by a degenerate denominator; it is not answered
here.

### Contrast with the correlative circuit

The other DAS thread found a direction that looked like a shared feature across two behaviours and
turned out to be a `neither` axis, because both behaviours put `neither` on the donor side.
**This case has no such confound**: the five configurations differ in intervening material and
share only the number contrast itself, which is the variable under study. Sharing the variable is
what a transfer test is for; sharing an incidental element is what invalidated the other one.
