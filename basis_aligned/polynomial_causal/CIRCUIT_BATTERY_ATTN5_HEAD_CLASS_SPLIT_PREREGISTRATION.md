# CIRCUIT BATTERY — ATTENTION 5 HEAD / CLASS SPLIT (preregistration)

Registered 2026-09-04 05:20Z (box clock; the draft said 05:22Z, two minutes ahead of the clock actually read -- corrected before freezing, and recorded rather than silently fixed). Claude, LANE 1 CUDA.
Rung `circuit_battery_attn5_head_class_split`. Script: `ops/circuit_battery_attn5_head_class_split.py`.
Input receipt: `circuit_battery_v2_results.json` (§2817, sha 5924b2549d…). IMMUTABLE.

## Object

§2829 and §2830: attention 5 is the type gate for six unrelated answer classes and is 20.4× more expensive per unit of its own write
than the median component on natural documents. Both are whole-component statements. This rung goes one level finer on the class gate
the same way §2820 went finer on the item writer: attention 5's write is `c_proj(concat_h o_h)` with no bias, so it decomposes exactly
and additively into nine head writes, each carried through the §2808 residual path-patching instrument and removed from every
downstream reader plus the direct path. The metric is candidate-class mass, with the margin recorded alongside so the two head maps can
be compared.

Fixed before the run: layer 5, all nine heads, behaviours = §2817's capable attn8-writer set, split OOD, family A1 only (this is a
localisation, not a selectivity claim). Zero fitted parameters. Sign convention: class-mass damage d_c = logmass_NATIVE − logmass_arm in
NATS, POSITIVE = the arm REMOVES class mass; margin damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS.

## Predictions

```
BARS  = {exact_tol: 1e-3, top2_share: .60, shared_pairs: 4, overlap: .50, top3_share: .80, floor: .05}
NULLS = {top2_share_le: .35, shared_pairs_le: 1, overlap_ge: .90, top3_share_le: .50}
```

**pred_a_head_decomposition_is_exact** — max over rows of `|Σ_h W_h − W|` ≤ 1e-3. *Worked example:* `c_proj` is linear without bias, so
the hypothesis reads fp32 round-off ~1e-6. The bar is 1e-3, not §2820's 1e-4: that section's identical check FAILED at 1.83e-4 purely on
fp32 accumulation over nine masked 1152-dimensional projections, and the bar is now set from the dtype rather than copied.

**pred_b_two_heads_carry_the_class_gate** — median over behaviours of `(class damage of head #1 + head #2) / max(whole-write class
damage, .05 nats)` ≥ .60. *Worked example:* §2820 found two of nine heads carry .877 of attention 8's write; if the class gate is
similarly concentrated, .6–1.0, and if all nine heads gate a little, 2/9 ≈ .22. Denominator floored at .05 nats. Null: ≤ .35.

**pred_c_the_class_head_set_is_shared** — the same unordered head pair is top-2 on at least 4 of the ≤7 behaviours.
*Worked example:* the answer classes differ (digits, roman numerals, month names, a repeated word), so a shared pair would mean one
fixed pair of heads gates "a member of the salient class" regardless of what the class is — the head-level version of §2829's
component-level finding. If each class recruits its own heads, the modal pair appears once or twice. Null: ≤ 1.

**pred_d_class_heads_differ_from_margin_heads** — median over behaviours of the fraction of the top-3 class heads that are also top-3
margin heads ≤ .50. *Worked example:* §2829's component-level version of this FAILED (overlap .667) because attn5 leads both maps, and I
argued there that attn5 is simply upstream of both jobs. At head level the two jobs can separate even if the component cannot: if
different heads of attn5 gate the class and move the margin, the overlap is 0/3 or 1/3. If it reads 3/3, the same heads do both and
attention 5 is one undivided mechanism. Fraction of a fixed count of three. Null: ≥ .90.

**pred_e_class_gate_heads_are_few** — median over behaviours of the top-3 heads' share of the whole write's class damage ≥ .80.
*Worked example:* a concentrated gate reads .85–1.0 (with the remaining six heads near zero or slightly negative, as attention 8's did
in §2820); a distributed one reads ~.35. This is the clause that decides whether "attention 5 gates the class" can be sharpened to
"three heads of attention 5 gate the class". Null: ≤ .50.

## Stated null

The class gate is spread across attention 5's heads (top-2 ≤ .35, top-3 ≤ .50), the pair is idiosyncratic per behaviour, and the same
heads carry the margin. That would say attn5 is a genuinely component-level gate with no finer structure, which is itself worth
recording given that §2820 found attention 8's write concentrated in two heads.

## Price

≤ 7 behaviours × (1 decomposition + 1 native + 1 whole + 9 head arms) per length-batch of 16 OOD rows.
Literal budget: ≤ 700 GPU forwards, 0 backwards, **0 fitted parameters**, < 2 GPU-minutes.

## What this does NOT claim

Head granularity only — no QK/OV decomposition and no per-position analysis of where attention 5 reads from. Target family only, so
nothing here is a selectivity claim (and §2820's lesson applies: a selectivity ratio without an admissibility gate crowns inert arms).
Class mass is defined by the bank's answer vocabularies. Does not satisfy Codex's four-phase integration contract; updates no circuit
record.
