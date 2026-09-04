# CIRCUIT BATTERY — IS ATTENTION 5's CE PRICE THE PRICE OF THE CLASS GATE? (preregistration)

Registered 2026-09-04 05:17Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_attn5_class_gate_price`. Script: `ops/circuit_battery_attn5_class_gate_price.py`.
Input receipt: `circuit_battery_class_mass_localisation_results.json` (§2829, sha b4d83e6790ab83eeb088faf53a4b5d24133244cf43eed40400aa708028a73793).
IMMUTABLE: any change gets a new document, not an edit.

## Why this is the top candidate

§2829 found that attention 5 is the type gate: ablating its write costs more candidate-class mass than any other component on 6 of 7
behaviours, across classes as unlike each other as single-token digits, roman numerals, month names and a repeated word. Independently,
and for weeks, the frontier lane has carried "attn5's write = the price cliff" as one of the three largest gaps in the explained
fraction. Two lanes reached the same component from opposite directions and have never been compared. This rung asks whether they are
the same fact.

Method: ablate each of the 36 components' write at EVERY position on natural documents and measure CE damage in nats; then correlate
that map, across components, with §2829's class-mass map and, as the discriminating control, with §2829's margin map. If class gating
is what attn5 is paid for, the class map should predict the CE price better than the margin map does.

Sign convention, stated because two conventions are in play: here d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. This is a
LOCAL ABLATION quantity on documents; it is NOT the §312 frontier's L2 (which is CE added above the real model by an installed
approximation, lower is better, frontier norm-2304 at 2.6735), and **nothing in this rung installs into that frontier or may be quoted
as an L2 number**. §2135 is not at issue because no approximation is being installed.

## Predictions

```
BARS  = {attn5_top: 3, rho_class: .50, dispro: 2.0, beats_margin: .10, ce_tol: .01}
NULLS = {attn5_top_ge: 8, rho_class_le: .10, dispro_le: 1.0, beats_margin_le: 0.0}
```

**pred_a_attn5_leads_the_document_ce_map** — attention 5 ranks in the top 3 of the 36 components by document CE damage.
*Worked example:* if the frontier lane's price cliff is a real property of this component, ablating it is among the most expensive
single-component ablations available, so rank 1–3; if the cliff is an artifact of the specific approximations that lane installs rather
than of the component, attn5 sits mid-table at rank 10–20. Integer rank in [1, 36]. Null: rank ≥ 8.

**pred_b_class_mass_predicts_the_ce_price** — Spearman correlation across the 36 components between §2829's pooled class-mass damage
and this rung's document CE damage ≥ .50. *Worked example:* if what a component is paid for is keeping probability on the contextually
appropriate class of next tokens, the two maps rank components alike, ~.5–.8; if class gating on my seven narrow behaviours has nothing
to do with general-document CE, ~0. Rank correlation over 36 paired values; no ratio, no floor needed. Null: ≤ .10.

**pred_c_attn5_is_disproportionately_expensive** — attn5's CE damage per unit of its own mean write norm, divided by the median
component's, ≥ 2.0. *Worked example:* "price cliff" means expensive beyond its size; a component that is costly merely because it
writes a lot reads ~1.0, while a genuinely disproportionate one reads 2–10. Denominator is a median of non-negative per-norm values and
is floored; both operands are non-negative because CE damage from removing a useful component is positive (a component whose removal
HELPS would make this negative, which would itself be reportable). Null: ≤ 1.0.

**pred_d_class_beats_margin_as_predictor** — (Spearman of class map vs CE) − (Spearman of margin map vs CE) ≥ .10.
*Worked example:* this is the discriminating clause. §2829 measured both maps on the same arms and found attn5 leads both, so the two
correlations will both be positive; the claim is that the CLASS map is the better predictor of the CE price, by at least .10. If the
margin map predicts as well or better, then what attn5 is paid for is not specifically class gating and this rung's headline collapses
to "attn5 matters", which is already known. A DIFFERENCE of two correlations, not a ratio. Null: ≤ 0.

**pred_e_instrument_reproduces_module_ce** — |manual-forward CE − the model module's own CE| ≤ .01 nats on the first chunk.
*Worked example:* the two are the same computation; float differences give ~1e-4. A larger gap means my document forward is not the
model and nothing else in the rung can be read.

## Stated null

attn5 is mid-table on documents, the class map does not predict the CE price, the cost is proportional to its write size, and the margin
map predicts as well. That would say the two lanes' interest in attn5 is coincidental, which is worth knowing before either lane spends
more on it.

## Price

37 ablation settings (36 components + native) × 32 natural documents of 256 tokens, in chunks of 8.
Literal budget: ≤ 200 GPU document-forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes.

## What this does NOT claim

Correlation across 36 components is not a mechanism, and a positive pred_b would license "class gating predicts what components cost",
not "attn5's CE price IS its class gate". The class map comes from seven narrow behaviours with bank-defined candidate sets; the CE map
comes from natural documents. Whole-component ablation only. Nothing installs into the §312 frontier. Does not satisfy Codex's
four-phase integration contract; updates no circuit record.
