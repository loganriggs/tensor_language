# Circuit battery — winner's curse precision replication preregistration

Registered 2026-09-04T08:33Z (`date -u` read in the same tool call that composed this header, per §2858's rule). Before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why, and what is deliberately NOT changed

§2864 selected the most-selective component on FIT and evaluated it on TEST. Two clauses failed, and both failed **into the undecided
zone between their own bar and their own null**:

| clause | measured | bar | null |
|---|---|---|---|
| pred_a — selection is inflated | **.063** | ≥ .15 | ≤ .05 |
| pred_b — the pick does not beat the named writer | **−.020** | ≥ 0 | ≤ −.05 |

A measurement that lands between its bar and its null has decided nothing, and §2865 has just demonstrated that this family of designs
is genuinely sample-size sensitive: §2863's pred_e flipped from .556 to **.609** at 2.7× the rows, while its pred_c moved the *other*
way (71.4% → 64.3%). So "more data" is not a formality here — it separated a noise miss from a real one once already.

Same design at **PER_CELL=24** (1.5× §2864's 16), everything else identical, **every bar carried over verbatim**. §2864's resolved
clauses — pred_c (−.266, selection beats random) and pred_d (0 of 7 argmin agreement) — are re-run unchanged as instrument checks: if
either moves materially, the replication is suspect and pred_a/pred_b cannot be read from it.

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
