# Circuit battery — winner's curse hold-out preregistration

Registered 2026-09-04T08:26Z (`date -u` read in the same tool call that composed this header — the rule from §2858, after four wrong
stamps in one hour). Before the run. Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Why

§2862 found, post-hoc, that the most-selective component under the v6 metric agrees between SELECT and TEST on **0 of 7** behaviours,
while the component RANKING reproduces at Spearman **.596**. That is a winner's curse: the minimum of a ratio over 36 candidates picks
whichever component's noise was most favourable on the split it was chosen on. §2862 reported it as exploratory and forbade the campaign
from reading "the most selective component" off a single split. This rung measures the effect honestly rather than avoiding it —
**select on FIT, evaluate on TEST** — and separates two facts that a single number would conflate: how badly selection is inflated, and
whether selection carries any signal at all.

Splits FIT (selection) and TEST (evaluation) are disjoint by the bank's split policy. The admissibility gate is §2820's (`|d_A1| ≥ .10`).

## Predictions, each with its worked-example line

Sign reminder: `selectivity = |d_C|/max(d_A1,.5)`, **LOWER = MORE SELECTIVE**, so "held-out MINUS selection" is **positive when
selection was inflated**, and "pick MINUS comparator" is **positive when the pick is WORSE**.

- **pred_a — selection is inflated.** median over behaviours of `selectivity_TEST(pick) − selectivity_FIT(pick)` ≥ **.15**.
  *Worked example:* if the FIT minimum is largely noise, a pick scoring .02 on FIT regresses to ≈ .25 on TEST, giving ≈ **+.23**; if
  selection is honest the two agree and it reads ≈ **.00**.
- **pred_b — the selected component does not beat the named writer on held-out data.** median `selectivity_TEST(pick) −
  selectivity_TEST(attn8)` ≥ **0**. *Worked example:* if the search is finding noise, its pick is no better than the component the
  battery identified from causal evidence, ≈ **+.05 to +.2**; if the search is finding real structure the battery missed, this goes
  **negative** and the battery's writer identification is what needs revisiting. Registered as the clause that could overturn the
  writer identification, not merely confirm the curse.
- **pred_c — selection still carries signal against a random component.** median `selectivity_TEST(pick) − median selectivity_TEST(12
  random live components)` ≤ **−.05**. *Worked example:* inflated but informative selection still beats a coin flip, ≈ **−.15**;
  worthless selection reads ≈ **.00** or positive. pred_a and pred_c are deliberately compatible: selection can be badly inflated AND
  still better than random, and that is the most likely outcome.
- **pred_d — the argmin does not reproduce.** FIT argmin equals TEST argmin on ≤ **2** of the behaviours. *Worked example:* §2862 saw
  0 of 7 on a different split pair; a stable argmin would give ≈ 5–7 and would mean §2862's caution was a split-specific accident.
- **pred_e — inert components are gated out.** at least one component excluded by `|d_A1| ≥ .10`, as in §2862.

## Nulls

- `a_null_selection_is_honest` (inflation ≤ .05) — no winner's curse; §2862's post-hoc observation does not survive a FIT/TEST design.
- `b_null_selection_beats_the_named_writer` (≤ −.05) — **the search finds a better component than the battery's identified writer, on
  held-out data.** That would not be a null in the ordinary sense: it would make the writer identification the thing to revisit, and
  it is registered here so that outcome is recognised rather than explained away.
- `c_null_selection_is_worthless` (≥ 0), `d_null_argmin_reproduces` (≥ 5 of the behaviours).

## Price

≤ 5,000 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 120 GPU-seconds. PER_CELL=16 as in §2862. Receipt:
`circuit_battery_winners_curse_holdout_results.json`, read with `price` in the same command the ledger section is written from, in the
canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
