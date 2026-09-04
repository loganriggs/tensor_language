# CIRCUIT BATTERY — WRITE RANK CENSUS (preregistration)

Registered 2026-09-04 05:32Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_write_rank_census`. Script: `ops/circuit_battery_write_rank_census.py`.
Input receipt: `circuit_battery_attn5_heldout_surrogate_results.json` (§2833, sha a474688fba25cdfdcac1e4b87518efc949a51ccaf2284ae4437d095a122706c0).
IMMUTABLE: any change gets a new document, not an edit.

## Why this is the control §2832 and §2833 need

§2832 and §2833 read as statements about attention 5: its write is 98.1% one direction, that direction is universal across corpora
(|cos| .997 natural vs code), and a held-out rank-32 basis reproduces 97.8% of its 2.20-nat value. **Neither section has an
across-component control**, so neither can distinguish "attention 5's write is remarkably low-rank" from "every write in this model is
low-rank, and attention 5 is simply the expensive one". This rung measures the same two quantities — top-direction energy and held-out
rank-32 surrogate cost — on all 36 components, with the basis fitted on 24 documents and scored on 24 DISJOINT ones.

Sign convention: d_ce = CE_arm − CE_NATIVE in nats, POSITIVE = the arm HURTS. **Not the §312 frontier's L2 (CE added above the real
model by an installed approximation, LOWER IS BETTER, frontier norm-2304 at 2.6735, §2135); nothing installs; the rank arms are
DIAGNOSTICS and metric-constructed bases/spans remain CLOSED (§2118 lineage).**

## Predictions

```
BARS  = {median_e1: .50, attn5_rank_top: 6, median_rank32: .10, rho: .50, ce_tol: .01}
NULLS = {median_e1_le: .20, attn5_rank_ge: 20, median_rank32_ge: .50, rho_ge: .80}
```

**pred_a_low_rank_writes_are_architectural** — the median over the 36 components of the fraction of write energy in the top singular
direction is ≥ .50. *Worked example:* if near-rank-1 writes are a property of this architecture, most components read .5–.99 and
attention 5's .981 is a high value in a high distribution; if attention 5 is exceptional, the median sits at .05–.2 and its .981 is an
outlier. **This prediction is registered in the direction that DEFLATES my own last two sections**, which is why it is first. Null: ≤ .20.

**pred_b_attn5_is_extreme_not_unique** — attention 5 ranks in the top 6 of 36 by top-direction energy. *Worked example:* it measured
.981; if writes are generally low-rank but attn5 is at the extreme, rank 1–6; if it is merely typical, rank 15–25. Integer rank in
[1, 36]. Note this pred and pred_a are deliberately complementary: pred_a says the phenomenon is general, pred_b says attn5 is still at
the top of it, and BOTH can be true — that is the outcome I expect and it is the one that most changes how §2832/§2833 should be read.
Null: rank ≥ 20.

**pred_c_cheap_surrogates_are_general** — the median over components of the held-out rank-32 surrogate's CE damage ≤ .10 nats.
*Worked example:* §2833 measured .049 for attention 5; if a 32-direction approximation is generally adequate, the median lands .01–.10,
and if attention 5 was unusually compressible, .3–1.0. Damages in nats, no ratio. Null: ≥ .50.

**pred_d_expensive_is_not_hard_to_approximate** — Spearman across the 36 components between ablation damage (`zero_damage`) and
held-out rank-32 damage ≤ .50. *Worked example:* if the components that cost the most to delete are also the ones a low-rank surrogate
fails on, the two maps rank alike at .7–.9 and cheap surrogates are useless exactly where they would matter; if cost and
approximability are unrelated, ~.0–.4. This is the clause that decides whether "writes are low-rank" is a useful fact or a vacuous one.
Rank correlation over 36 paired values. Null: ≥ .80.

**pred_e_instrument_reproduces_native_ce_matched** — |manual forward CE − the model module's own CE on the SAME chunk| ≤ .01 nats.
*Worked example:* the same computation on the same data, ~1e-4. Written in the matched-sample form that §2832's version of this check
got wrong.

## Stated null

Writes are generally high-rank (median top-direction energy ≤ .20), rank-32 surrogates are generally expensive (≥ .50 nats), attention 5
is ordinary in rank (≥ 20th), and the expensive components are precisely the hard-to-approximate ones (ρ ≥ .80). That would make
§2832/§2833 a genuinely attn5-specific finding — the opposite of what pred_a expects — and would be recorded as such.

## Price

36 components × (3 basis-fitting chunks on 24 documents + 3 zero-arm chunks + 3 rank-arm chunks on 24 disjoint documents), plus native
and instrument passes. Literal budget: ≤ 1,000 GPU document-forwards, 0 backwards, **36 × 32 × 1152 = 1,327,104 declared fitted
parameters** (one rank-32 basis per component, fitted on the fit documents only). < 5 GPU-minutes.

## What this does NOT claim

One rank (32) and one corpus (natural), with the held-out set a different slice of the same frozen cache — §2833 already showed the
cross-corpus version for attention 5 and this rung does not repeat it for all 36. Whole-write surrogates, no per-position adaptivity.
Nothing installs and no L2 numbers. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
