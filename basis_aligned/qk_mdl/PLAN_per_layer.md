# Per-layer decomposition master plan (audited 2026-07-30)

Goal: complete the four ledgers (Representation / Substitutability / Function / Meaning) for EVERY layer
1..17 bottom-up. Legend D=done P=partial M=missing. Full evidence in the audit (LOG tick 2026-07-30af+).

## Whole-model backdrop (covers every layer at once)
- **Representation** DONE for all 17 layers by the architecture identity (MLP composed fold + quartic
  pattern gauges, gates ~1e-7; §33). §38: two QK branches genuinely two-factor per head everywhere.
- **Substitutability** whole-model: PCA-64/head bottleneck +0.0475 (nulls 20-100×); analytic 18-MLP
  chain +0.0329 = 99.8% of the mean-ablation floor; pattern symbol-fold beats token-tables + random null
  at all layers 2-17 (§32, qk_l217_symbolgen).
- **Function**: 3 families (category engine MLP0-3, induction fabric, layout); census v2 = 30/162
  programmatic heads; attend-vs-predict map (selection_function_map.md).
- **Meaning**: boundary measured at 4 sites only — L0 (§34), block-3 category, L8 successor, L13 opener;
  plus induction MATCH predicate (held-out 98-111%).
- **Standing correction (§12q)**: zero-ablation inflated layer loads 10-60×; positional-mean floor
  measured ONLY for layers 2-5. Honest content-function of L2-5 patterns is 0.02-0.04 nats each.

## Status table (layers 1-17)
| L | Repr | Subst | Function | Meaning |
|---|---|---|---|---|
| 1 | D | D (pattern 99% token-identity; MLP1 fold 99.5%) | D (9-head ledger; MLP1=hub) | **P** (archetype names, NO code-gate; content untested) |
| 2 | D | D (sym+0.018 beats mean+0.027→~34% content; MLP2 fold 93.9%, prog 67.6%) | P (L2H5 induction core; 6/9 heads unmapped) | P (MATCH gated; category dial not load-bearing) |
| 3 | D | D (MLP3 chain 98.1%, prog 59.5%) | P (L3H8 strongest head, steer-confirmed; rest unmapped) | P (anti-self match gated) |
| 4 | D | D (sym TIES mean floor; MLP4 prog 82.5% but null-caveat) | **M** (ZERO programmatic heads; first dark layer) | **M** |
| 5 | D | D/P (sym LOSES to mean +0.065v0.036 — only fold loss; MLP5 prog 39.5%) | P (L5H7 general workhorse, not compound) | **M** |
| 6 | D | P (no mean floor/SE; MLP6 NO-substitution −32%) | P (L6H5 MATCH_prev) | **M** |
| 7 | D | P (no mean floor/SE; MLP7 prog 30%) | P (L7H3 induction carrier) | **M** |
| 8 | D | P (no mean floor/SE; MLP8 prog 24%) | P (largest successor-cache reader) | P (successor table, held-out fails) |
| 9 | D | P (MLP9 prog 21%) | P (L9H8 KEY_newline, anchor FALSIFIED, open) | **M** |
| 10 | D | P (MLP10 prog 20%) | P (1 programmatic head) | **M** |
| 11 | D | P (MLP11 prog 24%) | P (L11H4 KEY_newline divergent) | **M** |
| 12 | D | P (MLP12 prog 16%, worst) | P (L12H6 MATCH_prev→newline) | **M** |
| 13 | D | P (MLP13 prog 25%) | P (L13H2/H8) | P (opener flag, gated, d=−1.31) |
| 14 | D | P (MLP14 prog 20%) | P (L14H6 MATCH_prev) | **M** |
| 15 | D | D/P (best symbol-expressibility; MLP15 prog 69%) | P (L15H3/H4 KEY_cap clean cluster) | **M** (KEY_cap best un-named candidate) |
| 16 | D | D/P (MLP16 prog 94.5%) | P (4 programmatic heads; MLP16 lexical) | **M** |
| 17 | D | D (MLP17 largest module, prog 89.3%) | **P** (ZERO programmatic heads; output readout) | **M** |

## Corrections to the old ROADMAP claim
1. Substitutability near-complete TRUE, two holes: positional-mean floor unmeasured for L6-17; per-layer
   MLP data-fit surrogates weak in L5-14 band (chain covers causally, no compression win). No per-token SEs.
2. "Function patchy above L8" PARTIALLY SUPERSEDED by census v2 + attend-vs-predict; real holes are L4 &
   L17 attention (dark), mid-stack MLP band 4-15 (no family), KEY_newline mechanism, 132 non-prog heads.
3. Meaning is the frontier, understated: only 4 sites; L1-2 names ungated; L4-7,9-12,14-17 nothing.

## Prioritized experiments (bottom-up; layers 1-3 first)
1. **L1 content-nameability gate** (Meaning, frontier root) — §34 protocol on layer-1 head-value spectra
   + stage23 archetypes, substitution-gated FW[448:600]. Adapt qk_coord_semantics_l0/arch with
   mean-residual tables. Is "content spectral" a L0 quirk or the rule?
2. **L1 selection-name gate** (Meaning, L1) — code the 9 validated L1 archetype clusters as predicates,
   census-style simultaneous-substitution gate + SEs. Adapt qk_selection_census_v2 gate stage at L1.
3. **Positional-mean floors + SEs for L6-17 patterns** (Subst hygiene, cheapest way table is wrong) —
   add mean-pattern arm (§12q) + paired per-token SEs to qk_l217_symbolgen and rerun.
4. **L2/3 composed-fold interfaces + program gap** — MLP3 one-hop fold (adapt qk_l2_composed_fold),
   per-layer chain-vs-program numbers + SEs, so L1-3 each carry one defensible substitutability cell.
5. **L4 function census** (first dark layer) — knockout battery (adapt qk_circuit_atlas) on L4's 9 heads
   + MLP4 + predicate-library extension; extend templates or conclude "diffuse" with evidence.
6. **Mid-stack MLP family assignment MLP4-15** — per-MLP knockout vs task battery + category trajectory
   (adapt qk_category_engine); one script, 12 rows, closes the largest Function hole.
7. **Selection scaffold stages 1-3 for L2-5** — extend qk_l1_stage1/stage23 using existing mean-residual
   ports (qk_l2_tables.pt…qk_l5_tables.pt); per-head archetype vocabularies to feed the gate.
8. **KEY_newline mechanism resolution** (L9/11/13/16 + L2) — test clause-position / sentence-initial /
   punct-context hypotheses with the damage-split design; gate any surviving name.
9. **KEY_cap capitals code, L15-16** (5th meaning site) — name capitalization predictor as code, §35
   protocol (name→gate→dial→extract→red-team) on FW[448:600]. Adapt algo_tasks/semantics_opener pipeline.
10. **L5 content account** — the one fold-loss layer; re-fit with L5-specific symbols, characterize L5H7.
11. **(Capstone) Per-layer master substitutability table with SEs** — one script, each layer's attention
    fold + best MLP surrogate simultaneously, cumulative, base-relative dCE + SEs vs mean-floor + uniform ceiling.

## Driver results (qk_layer_decomp.py, 3 ledgers per layer; Meaning via separate gates)
Substitutability = marginal dCE of replacing layer L's attention (PCA-64/head bottleneck) + MLP
(composed fold), held-back FW[448:600], paired standard error; null = head-span random basis.
| L | Repr gauge | Subst dCE ± SE | null (×margin) | %uniform-ceiling | programmatic heads |
|---|---|---|---|---|---|
| 1 | 1.2e-6 | +0.00052 ± 0.00016 | 0.025/0.020 (40-48×) | 99.99% | H3 PREV1(.17), H4 MATCH_same(.15) |
| 2 | 9.8e-7 | +0.00136 ± 0.00020 | 0.0055 (4×) | 99.98% | H4 KEY_punct(.06), H5 MATCH_same(.25 induction core) |
| 3 | 9.0e-7 | +0.00093 ± 0.00017 | 0.005 (5.4×) | 99.99% | H5 MATCH_same(.13), H8 MATCH_same(.31 strongest head) |
| 4 | — | +0.00274 ± 0.00027 | 0.0091 (3.3×) | 99.96% | **NONE** — DIFFUSE (all 9 heads gain ≤0.027; confirms first dark layer) |
