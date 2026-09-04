# CIRCUIT BATTERY — CONSTANT WRITE CENSUS (preregistration)

Registered 2026-09-04 05:44Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_constant_write_census`. Script: `ops/circuit_battery_constant_write_census.py`.
Input receipt: `circuit_battery_attn5_direction_identity_results.json` (§2835, sha 1424ab0c93f5560009b5c64b206e6f734e6a79bbbca179db79eb4f2373b4e4ed).
IMMUTABLE: any change gets a new document, not an edit.

## Object

§2835 showed attention 5's write is a constant to measurable tolerance: its top singular direction IS its mean write (|cos| .9999996),
its gain has coefficient of variation .080 and never changes sign, and replacing it with one fixed vector costs .1286 nats of a
2.211-nat component — the entire per-position gain being worth .0022. §2834's census covered top-direction ENERGY for all 36 components
but not the two properties that actually make a write a constant. This rung measures those everywhere, plus the CE cost of replacing
each component's write outright with its own mean vector fitted on 24 documents and scored on 24 DISJOINT ones.

The output is an enumeration of which parts of bilin18 are, to measurable tolerance, biases — the cheapest possible entries in a
compiled tensor program, and the parts of the model that need no computation at all.

**Constant-like** is defined here, before the run: `|cos(top singular direction, mean write)| ≥ .90` AND `gain CV ≤ .50`.

Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not the §312 frontier's L2 (CE added above the real
model by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; DIAGNOSTICS only and
metric-constructed bases/spans remain CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {n_constant: 4, recover: .80, attn5_cv_top: 3, early_frac: .60, ce_tol: .01}
NULLS = {n_constant_le: 1, recover_le: .40, attn5_cv_rank_ge: 12, early_frac_le: .30}
THRESHOLDS = {cos: .90, gain_cv: .50}
```

**pred_a_attn5_is_not_alone** — at least 4 of the 36 components are constant-like. *Worked example:* §2834 found the median
top-direction energy is only .321, so most writes are not one-dimensional at all; but several sat high in that ordering (attn6, attn1,
attn7, mlp15, attn9, mlp16, attn10), and if being one-dimensional usually means being constant, 4–10 components qualify. If attention 5
is a one-off, 0–1 do — which would make §2835 a fact about a single component rather than a structural feature. Count in [0, 36].
Null: ≤ 1.

**pred_b_constant_arms_recover_their_components** — the median over the constant-like components of
`1 − d_ce(CONST)/d_ce(ZERO)` ≥ .80. *Worked example:* attention 5 recovered .943; if the constant-like test picks out components that
really are biases, the median lands .8–.95, and if the test is loose it lands .3–.6, meaning cos and CV are the wrong screen. Both
operands are damages in the same units; the denominator is a deletion cost and components with near-zero deletion cost produce a NaN
ratio, which is excluded from the median and counted rather than treated as zero. Null: ≤ .40.

**pred_c_attn5_has_the_steadiest_gain** — attention 5 ranks in the top 3 of 36 by lowest gain CV. *Worked example:* it measured .080;
if constancy is what makes it the price cliff, it should be at or near the extreme, rank 1–3. If it sits at rank 15–25 then other
components are steadier and being a constant is not what distinguishes it. Integer rank in [1, 36]. Null: ≥ 12.

**pred_d_constant_writes_are_early** — at least .60 of the constant-like components sit at layer ≤ 8. *Worked example:* a constant
offset is most useful before the layers that read it, and §2830's most expensive components (mlp0, attn1, attn5, attn0, mlp1) are all
early; so if constancy and earliness go together, .6–1.0. If constant-like components are spread evenly across depth, ~.5, and if they
cluster late, ≤ .3. A fraction of the constant-like set; reported as NaN and flagged if that set is empty. Null: ≤ .30.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats, in
the matched-sample form.

## Stated null

Attention 5 is alone (≤ 1 constant-like component), the screen does not identify real biases (median recovery ≤ .40), attn5 is not
especially steady, and any constants there are sit late. That would confine §2835 to one component and remove the compilation reading.

## Price

36 components × (3 fit chunks on 24 documents + 3 zero chunks + 3 const chunks on 24 disjoint documents), plus native and instrument
passes. Literal budget: ≤ 1,000 GPU document-forwards, 0 backwards, **36 × 1152 = 41,472 declared fitted parameters** (one mean vector
per component). < 5 GPU-minutes.

## What this does NOT claim

One corpus (natural), one held-out slice of the same frozen cache; §2833 did the cross-corpus check for attention 5 only. Replacing a
write with a constant is measured ONE COMPONENT AT A TIME — this rung says nothing about replacing several at once, which is where
interactions (and §2818's super-additivity) would appear. Nothing installs; no L2 numbers; the constant arms are diagnostics of
compilability. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
