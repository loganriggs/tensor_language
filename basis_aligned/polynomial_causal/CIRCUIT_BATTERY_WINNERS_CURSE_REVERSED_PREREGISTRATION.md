# Circuit battery — winner's curse reversed-direction preregistration

Registered 2026-09-04T08:47Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why, and what is deliberately NOT changed

§2869 selected on FIT and evaluated on OOD, and selection beat a random live component by **−.240** — against only **−.088** when
evaluated on TEST at the same sample size (§2867). A held-out-**vocabulary** population separated the selected component from a random
one *more* cleanly than the nearer TEST population did. §2869 recorded that as unexpected and deliberately did not explain it.

Two accounts fit. Either OOD is doing the work **as an evaluation population** — it spreads components apart, so any real difference
shows up larger — or the effect is **directional**, tied to selecting on FIT specifically. Reversing the roles distinguishes them:
**select on OOD, evaluate on FIT.**

If the advantage stays near −.24, the population is doing the work and direction does not matter. If it collapses toward §2867's −.088,
the asymmetry was directional and **§2869's −.240 must not be read as a property of the metric** — which is the outcome that would
bound my own §2869 and is registered so it can be recognised.

PER_CELL=24 as in §2867 and §2869, so all three are directly comparable; **every bar carried over verbatim**.

## Predictions

Verbatim from `CIRCUIT_BATTERY_WINNERS_CURSE_PREREGISTRATION.md`. Sign reminder: `selectivity = |d_C|/max(d_A1,.5)`, **LOWER = MORE
SELECTIVE**; "held-out MINUS selection" is **positive when selection was inflated**; "pick MINUS comparator" is **positive when the pick
is WORSE**.

- **pred_a** — median `selectivity_TEST(pick) − selectivity_FIT(pick)` ≥ **.15**. *Worked example:* a pick scoring .02 on FIT that
  regresses to .25 on TEST reads **+.23**; honest selection reads ≈ **.00**. §2864 measured **.063**.
- **pred_b** — median `selectivity_TEST(pick) − selectivity_TEST(attn8)` ≥ **0**. *Worked example:* a search finding noise is no better
  than the causally-identified writer, ≈ **+.05 to +.2**; a search finding real structure the battery missed goes **negative**, and the
  writer identification is then what needs revisiting. §2864 measured **−.020**.
- **pred_c** — median `selectivity_TEST(pick) − median selectivity_TEST(12 random live components)` ≤ **−.05**. *Worked example:*
  inflated but informative selection reads ≈ **−.15**; worthless selection ≈ **.00** or positive. §2864 measured **−.266**.
- **pred_d** — FIT argmin equals TEST argmin on ≤ **2** behaviours. *Worked example:* §2864 measured **0 of 7**; a stable argmin gives
  5–7.
- **pred_e** — at least one component excluded by the `|d_A1| ≥ .10` gate (§2820).

## Nulls

Carried over verbatim: `a_null_selection_is_honest` (≤ .05), `b_null_selection_beats_the_named_writer` (≤ −.05) — which would make the
battery's writer identification the thing to revisit, and is registered so that outcome is recognised rather than explained away —
`c_null_selection_is_worthless` (≥ 0), `d_null_argmin_reproduces` (≥ 5).

## Price

≤ 12,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 240 GPU-seconds — 1.5× §2864's 6,408 / 75.3 s. Receipt:
`circuit_battery_winners_curse_precision_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
