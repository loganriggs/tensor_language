# Circuit battery — protocol v6 preregistration: P is a POSITIVE control, and scoring it as a negative one pinned the metric at 1

Registered 2026-09-04T08:15Z (`date -u` read immediately before composing this header, per the rule adopted in §2858), before the run.
Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## The error being corrected, and the documentary control that establishes it

§2858 repaired the answer-preserving control — it had been byte-identical to the target on 11 of 21 behaviours — found the corrected
ratio essentially unchanged, and read that as *"attn8's write is required for the surface form of the continuation, not for the causal
variable the behaviour turns on."* **That interpretation is wrong.** The control that establishes it is documentary, needs no GPU, and
comes from the bank's own declarations rather than from any measurement of mine:

| | declared value |
|---|---|
| `TASKS['paren_list.index_successor'].causal_variable` | **"last visible list label"** |
| P family `semantic_details` | **`{'perturbation': 'item_words'}`** |
| generator source comment at the P branch | **"surface: swap the item words only"**, "unrelated filler word changes" |

P swaps filler words and **leaves the causal variable exactly intact** — `64) jasmine / 65) vine` → `64) beacon / 65) mosaic`, labels
64/65 unchanged. It preserves the answer *because* it preserves the variable. Therefore a writer that genuinely carries the causal
variable **must** damage P as much as A1: `|d_P|/d_A1 ≈ 1` is the **signature of a variable-carrying writer**, not evidence against
selectivity. The score took `max` over controls, `|d_P|/d_A1 ≈ 1` always, so the metric was **pinned near 1 for every behaviour no
matter how specific the writer was**, and the `max` discarded the one control that carries information.

From §2858's own landed receipt (`circuit_battery_preserving_control_repair_results.json`), on the seven attn8 behaviours:

| | min | max |
|---|---|---|
| `\|d_P_donor\|/d_A1` (variable-preserving control) | .96 | 1.27 |
| `\|d_C3\|/d_A1` (copy control, answer NOT a function of the variable) | **.27** | **.63** |

and `d_C3` is **negative on all seven** — removing attn8's write *helps* the copy answer. So the campaign's standing negative, "zero
behaviours are writer-selective", is not a fact about the model: it is what a `max` over a positive control returns. §2840's and
§2852's verdicts are affected; this rung measures the replacement rather than asserting it.

## The re-specification

    selectivity      = |d_C| / max(d_A1, .5)                 -- copy control; its answer is NOT a function of the causal variable. LOWER = MORE SELECTIVE.
    positive control = |d_P_donor - d_A1| / max(d_A1, .5)    -- SMALL confirms the writer carries the variable.

Scored on **held-out splits (SELECT, TEST, OOD)**, which the selectivity stage has never used. The arm is §2852's calibrated ladder
rung per behaviour, **read from that receipt and frozen, not re-fit here**, so no parameter is chosen on the data being scored. The
bank is not mutated; the copy control is §2857's `v2_triple`, derived from frozen rows.

## Predictions, each with its worked-example line

- **pred_a — the two controls measure different things.** median |old-style ratio − new selectivity| ≥ **.30** on SELECT.
  *Worked example:* old-style takes `max(|d_P|,|d_C|)/d_A1` ≈ 1.0; if the copy control reads ≈ .45, the difference is ≈ **.55**. If P
  and C carried the same information the difference is ≈ **.00** and the re-specification is empty.
- **pred_b — P behaves as a positive control.** median `|d_P_donor − d_A1|/d_A1` ≤ **.25**. *Worked example:* a variable-carrying
  writer damages the variable-preserving prompt just as much, so ≈ **.05–.2**; a writer that did NOT carry the variable would leave P
  intact and read ≈ **1.0**, which would refute the whole reading above.
- **pred_c — the writer opposes the copy answer.** `d_C3 < 0` on ≥ **6 of 7** behaviours. *Worked example:* attn8 writes "which item
  was last and whether it is round" (§2842/§2843), which competes with verbatim copying, so removing it should *help* the copy answer:
  negative. A writer indifferent to the copy answer reads ≈ **0** with random sign.
- **pred_d — selectivity generalises.** median |selectivity(SELECT) − selectivity(TEST)| ≤ **.15**. *Worked example:* a real property
  of the writer transfers to disjoint held-out situations, ≈ **.05**; a number driven by the particular SELECT rows ≈ **.4+**.
- **pred_e — the behaviour ordering is stable.** Spearman ρ(selectivity SELECT, TEST) ≥ **.60**. *Worked example:* if selectivity is a
  per-behaviour fact the ranking survives the split, ρ ≈ **.8**; if it is noise, ρ ≈ **.0**. (Registered on the SHARED behaviour axis —
  the invariant index — not on a sample-indexed loading; that is the trap recorded against §2647/§2649.)

## Nulls

- `a_null_metrics_agree` (≤ .10): the re-specification changes nothing and §2858's reading survives.
- `b_null_preserving_control_is_not_positive` (≥ .50): **P is NOT a positive control**, which would refute this document's central
  claim and restore §2858's reading. This is the clause that can most cleanly prove me wrong and it is registered for that purpose.
- `c_null_writer_does_not_oppose_copy` (≤ 3 of 7), `d_null_does_not_generalise` (≥ .40), `e_null_ordering_is_noise` (ρ ≤ .20).

## Price

≤ 1,800 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 70 GPU-seconds. Receipt:
`circuit_battery_variable_dependence_selectivity_results.json`, read with `price` in the same command the ledger section is written
from, in the canonical `Price: N GPU forwards, X GPU-seconds` / `Results: <file>.json` form the guard can parse (§2853, §2858).
