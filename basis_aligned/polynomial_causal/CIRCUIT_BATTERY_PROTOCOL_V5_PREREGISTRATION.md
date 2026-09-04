# Circuit battery — protocol v5 preregistration: the answer-preserving control was the target condition itself

Registered 2026-09-04T08:0xZ, before the run. Immutable; the rung's frozen-hash check refuses to execute if this file changes.

## Provenance, stated plainly

**The defect was found before this document was written, in a smoke run of the withdrawn V4 rung.** V4 set out to re-score selectivity
with §2857's repaired copy control; its smoke output showed the A1 and P families reading identical native margins AND identical
damages on five of seven behaviours (keyed_line 2.38/2.38 native, 2.60/2.60 damage; numbered_list 2.07/2.07, .90/.90; paren_list
1.34/1.34, 1.00/1.00; roman_list 2.54/2.54, 1.92/1.92). Reading the bank confirmed the structural cause. So `n_identical ≥ 5` below is
a bar informed by a smoke observation, not a blind prediction, and it is registered as such. What is NOT yet measured, and is the
actual content of this rung: whether the identity is exact and universal across the full 21-task bank at the registered PER_CELL, what
the donor-side control reads instead, and whether the corrected score changes §2852's published verdict.

## The defect

The bank emits a group's four families as transformations of ONE situation, and the transformation lives in the **donor**: A1, A2 and
P share the same `base_text`/`base_answer`, and only C has a base of its own. Verified directly on `paren_list.index_successor`:

| family | base_text (tail) | base_answer | donor_text (tail) | donor_answer |
|---|---|---|---|---|
| A1 | `64) jasmine\n65) vine\n` | 66 | `16) jasmine\n17) vine\n` | 18 |
| P  | `64) jasmine\n65) vine\n` | 66 | `64) beacon\n65) mosaic\n` | 66 |

Every scoring rung to date calls `pack(b, "base")` for **every** family. The answer-preserving control was therefore evaluated on the
target's own prompts, giving `ratio = |d_P| / max(d_A1, .5) = 1.000` **by construction**. §2852 read exactly 1.00 on six of eight
behaviours and its docstring attributes that to both sides being saturated; if this rung's pred_a and pred_b hold, the real cause is
that numerator and denominator are the same measurement, and §2852's headline — ZERO behaviours writer-selective — was scored against
a control that cannot, even in principle, differ from the target.

Because that is a **conclusion-flipping correction**, the standing rule requires an independent physical control before publication.
One exists that needs no GPU and no model: **byte-identity of the prompt sets**. pred_a is that control; pred_b is its forced
numerical consequence. Neither can be produced by a bug in the measurement code, because pred_a never runs the model.

## The repair

The correct answer-preserving condition is the P family's **donor**: same answer, different causal variable. This rung scores five
conditions per behaviour with §2852's calibrated arm on attn8 — `A1` (base), `P_base` (the degenerate control, kept so the defect is
measured rather than asserted), `P_donor` (the repair), `C_base` (§2852's copy control), `C3` (§2857's repaired copy control, derived
from frozen rows) — and compares the old and corrected ratios. Arm ladder, ceiling (.80), usability bar (native ≥ .50), splits and
PER_CELL=24 are carried over from §2852 unchanged. **The bank is not mutated**, so every `FROZEN_ROW_HASHES` entry and every earlier
receipt stays reproducible.

## Predictions, each with its worked-example line

- **pred_a — the preserving control is byte-identical to the target.** ≥ 5 of the censused tasks have `frac_base_in_a1` = 1.00
  (every P SELECT base prompt appears in the A1 SELECT base set). *Worked example:* if the families were genuinely distinct
  situations this fraction is ≈ **0.00**; if P is a transformation stored in the donor it is exactly **1.00**. No model is run.
- **pred_b — the identity forces the damage to coincide.** On tasks with `frac_base_in_a1` = 1.00, max |d_P_base − d_A1| ≤ **.015**
  (registered CUDA-atomics tolerance). *Worked example:* identical prompts through identical arms must give **.000** up to atomics;
  anything at .1+ would mean the sets are not in fact identical and pred_a's census is wrong.
- **pred_c — the donor-side control is informative.** median over behaviours of |d_P_donor − d_A1| ≥ **.15**. *Worked example:* if
  attn8's write is specific to the causal variable, the donor (same answer, different item) should be damaged **less** than the
  target, giving a separation of a few tenths; if attn8 is simply required to emit any list continuation, the donor is damaged just as
  much and this reads ≈ **.00** — which would say the standing negative was RIGHT for a reason §2852 never actually established.
- **pred_d — the corrected score changes the verdict.** ≥ 1 behaviour crosses the ≤ .25 selectivity bar in either direction relative
  to §2852. *Worked example:* a behaviour at §2852's 1.000 whose corrected reading is max(|d_P_donor|, |d_C3|)/d_A1 = 0.3/1.9 reads
  **.16** and becomes selective; if every corrected ratio stays near 1, nothing crosses and the negative stands on a fixed instrument.
- **pred_e — the derived copy-control rows are valid.** 0 violations of: single-token answer, joint tokenization intact, answer ≠ its
  own group's A1 successor answer, answer token literally present in the prompt. *Worked example:* §2857 measured 0 on the same
  construction, so ≠ 0 here means the derivation drifted.

## Nulls (any one met = a negative reported on that clause)

- `a_null_no_task_is_degenerate`: 0 tasks byte-identical — the defect does not exist, §2852 stands untouched, and the identical
  smoke readings had some other cause I have not found.
- `c_null_donor_control_is_also_the_target`: median separation ≤ .05 — the repair changes the prompts but not the measurement.
- `d_null_verdict_unchanged`: nothing crosses the bar — **§2852's negative survives on a repaired instrument**, which is the outcome
  that would most strengthen the campaign's central claim, and is worth the price on its own.

## Scope note, registered in advance

`pack(b, "base")` appears in ~20 landed rung scripts. Most measure damage on A1 alone (localisation, lens, geometry) and are
unaffected — the defect only corrupts comparisons that treat P as a condition DISTINCT from the target. This rung does not audit those
sections; it establishes the fact. The audit is a separate backlog item and no claim is made here about which other §§ are affected.

## Price

≤ 1,400 GPU forwards, 0 backwards, 0 fitted parameters, ≤ 60 GPU-seconds. Receipt:
`circuit_battery_preserving_control_repair_results.json`, read with `price` in the same command the ledger section is written from
(§2853).
