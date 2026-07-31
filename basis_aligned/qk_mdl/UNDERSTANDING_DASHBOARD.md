# UNDERSTANDING DASHBOARD — bilin18, per-component (180 rows)

Assembled 2026-07-31 from existing artifacts only (no new experiments): PLAN_per_layer.md driver table,
qk_allterm_census.json (§89), qk_census_difficulty.json (§67), qk_unsup_positional.json (§62),
qk_selection_census_v2.json, qk_coverage_ledger.json (§71), RESULTS_l0_mdl.md §32–§95,
TECHNIQUES_unsup_discovery.md. Held-back slice FW[448:600] throughout.

## Legend — the five understanding levels
| level | question | source |
|---|---|---|
| **L1 REPRESENTED** | exact rewrite exists | architecture identity: MLP composed fold + quartic pattern gauges, gates ~1e-7 (§33); two-branch two-factor per head (§38); stream-term reconstruction ~1e-6 every layer (§89) — **Y for all 180 by construction** |
| **L2 SUBSTITUTABLE** | causal replacement through a compact interface | per-layer driver: marginal dCE of replacing the layer's attention (PCA-64/head) + MLP (composed fold), ± paired SE (PLAN_per_layer.md); every component inherits its layer's number. Layer 0 has no marginal driver row — covered by the whole-model chain (+0.0329 all-18-MLP, +0.0475 PCA bottleneck, §33) |
| **L3 ANATOMY** | parts and triggers characterized | MLPs: §89 all-term census (floor, terms-to-95%, top terms). Heads: verified type-detector characterization (§56–§64/§68 toolbox), census v2 selection predicate, or §62 positional label |
| **L4 MECHANISM** | a HOW statement in RESULTS | e.g. v1-router §41/§50, digit algorithms §65, differential pair §91, hub five-terms §86 |
| **L5 NAMED-ALGORITHM** | name passed a substitution/verification gate | strictest: induction MATCH predicate (§47/§49), L13H8 value-swap gates (§41/§50), digit heads (§63/§65), capital selector mlp.L17.d1 (§69/§75), §56 causally-verified discoveries, §68 class-push specificity gates |

Values: **Y** yes · **P** partial (marked liberally: predicate-only names, family-level gates, corrected/failed names) · **N** no.
Causal weight: heads = census global mean-ablation dCE (solo path, §67); MLPs = whole-block mean-ablation floor (§89).
Caveats (honesty): the two weight kinds are not the same granularity (a full MLP block vs one head path); solo importances
under-count combinations — whole-model super-additivity is 2.87x (§71) — so "causal-mass fraction" means fraction of SUMMED
SOLO mass, not of the joint headroom. Nine heads have slightly negative solo dCE (clipped to 0 for weighting).

## (b) Summary — coverage by level
Total causal mass (summed solo): **10.100 nats** = heads 0.631 + feed-forward blocks 9.470.

| level | components Y | comp-fraction Y | causal-mass Y | components Y+P | comp-fraction Y+P | causal-mass Y+P |
|---|---|---|---|---|---|---|
| L1 represented | 180/180 | 100% | 100.0% | 180/180 | 100% | 100.0% |
| L2 substitutable | 180/180 | 100% | 100.0% | 180/180 | 100% | 100.0% |
| L3 anatomy | 73/180 | 41% | 97.6% | 110/180 | 61% | 98.6% |
| L4 mechanism | 20/180 | 11% | 69.6% | 62/180 | 34% | 90.2% |
| L5 named-algorithm | 9/180 | 5% | 5.2% | 32/180 | 18% | 68.8% |

Reading: representation and substitutability are complete (that is the program's foundation); anatomy is broad;
mechanism covers a minority of components but a large share of causal mass (the early feed-forward blocks and the
readout have mechanism statements, and they carry most of the solo mass); strict gate-passing names cover only
9 components. Note the L5 causal-mass figure is dominated by the single row mlp.L17 (floor 0.421);
head-level gate-passing names carry only a small slice of head mass, and the L5 Y+P mass figure is dominated by
the P grades of mlp.L1 (five-term sufficiency gate, §86) and mlp.L2 — remove those two rows and Y+P mass drops to
~6%. These are SOLO-mass fractions — §71's honest
range for named coverage of the model's total causal headroom is ~11% (rising to ~44-46% of single-path-expressible
effect, §76), because most computation is combinational and below the path basis.

## (c) Top-10 most causally important components — level profile
| rank | component | weight (nats) | L1 | L2 | L3 | L4 | L5 | one-line status |
|---|---|---|---|---|---|---|---|---|
| 1 | mlp.L1 | 5.574 | Y | Y | Y | Y | P | THE HUB: five named stream-pair terms ARE the block (keep-5 = +0.0019 of a 5.574 floor, red-teamed, §86) |
| 2 | mlp.L0 | 1.234 | Y | Y | Y | P | N | category-engine member (§32/§44) |
| 3 | mlp.L2 | 0.739 | Y | Y | Y | Y | P | "SQUARE THE PREVIOUS BLOCK'S OUTPUT": MrxMr alone leaves 0.093 of the 0.739 floor |
| 4 | mlp.L3 | 0.616 | Y | Y | Y | P | N | category-engine member (§32/§44) |
| 5 | mlp.L17 | 0.421 | Y | Y | Y | Y | Y | DIFFERENTIAL PAIR readout: mlp-recent^2 writes a broad lexical-class prior, attention-earlier x mlp-recent ... |
| 6 | mlp.L4 | 0.148 | Y | Y | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 7 | mlp.L16 | 0.148 | Y | Y | Y | Y | N | PURE HISTORY-READER: MexMe + AexMe leave 0.033 of the 0.148 floor, own attention causally dead (§89) |
| 8 | mlp.L5 | 0.093 | Y | Y | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 9 | mlp.L6 | 0.083 | Y | Y | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 10 | h.L0.3 | 0.073 | Y | Y | Y | Y | Y | prev-token head (FIXED-OFFSET back-1, purity 0.69, §62 |

## (d) Priority list — MATTERS-MOST x UNDERSTOOD-LEAST
Components with high causal mass but no full mechanism statement (L4 != Y) and no gate-passing name (L5 != Y),
ranked by weight — the honest to-do list for future mechanism work:

| rank | component | weight | L3 | L4 | L5 | what is missing |
|---|---|---|---|---|---|---|
| 1 | mlp.L0 | 1.234 | Y | P | N | category-engine member (§32/§44) |
| 2 | mlp.L3 | 0.616 | Y | P | N | category-engine member (§32/§44) |
| 3 | mlp.L4 | 0.148 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 4 | mlp.L5 | 0.093 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 5 | mlp.L6 | 0.083 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 6 | mlp.L7 | 0.060 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 7 | mlp.L9 | 0.056 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 8 | mlp.L8 | 0.051 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 9 | mlp.L11 | 0.047 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 10 | mlp.L10 | 0.046 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 11 | mlp.L12 | 0.044 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 12 | mlp.L13 | 0.040 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |
| 13 | mlp.L15 | 0.039 | Y | P | N | direction d2 = verified punctuation->capital remap (drop 0.0068±0.0009, z 7.7, full control, §64)... |
| 14 | mlp.L14 | 0.030 | Y | N | N | mid-stack distributed refinement — no distinct family (§44) |

These 160 components carry 3.070 nats = 30.4% of the summed solo mass.
Structural context (§74/§76/§83): much of this is measured to be irreducibly distributed / static class priors,
so "understand it" here means sufficiency-style anatomy (the §86/§89 term route), not more single-path naming.

Because whole-block MLP floors dwarf single-head paths, the list above is all feed-forward; the same ranking
restricted to ATTENTION HEADS (the head-level frontier):

| rank | head | weight | L3 | L4 | L5 | status |
|---|---|---|---|---|---|---|
| 1 | h.L1.1 | 0.0296 | Y | P | N | self-attention head (FIXED-OFFSET back-0, purity 0.75) |
| 2 | h.L6.3 | 0.0262 | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.43, §62) |
| 3 | h.L7.8 | 0.0194 | Y | P | P | verified distributed SUBWORD class-pusher (specificity z 6.4, §76) |
| 4 | h.L9.7 | 0.0184 | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.57, §62) |
| 5 | h.L7.0 | 0.0170 | N | N | N | census fingerprint only (trigger/effect measured, §67) |
| 6 | h.L4.1 | 0.0140 | Y | P | N | FIXED-OFFSET back-1 (purity 0.40) |
| 7 | h.L11.6 | 0.0128 | P | N | N | weak positional signature (FIXED-OFFSET (back-1), purity 0.31, §62) |
| 8 | h.L4.5 | 0.0118 | P | N | N | weak positional signature (FIXED-OFFSET (back-1), purity 0.40, §62) |
| 9 | h.L5.7 | 0.0114 | Y | P | N | absolute-position-0 sink (purity 0.96) |
| 10 | h.L1.4 | 0.0109 | Y | P | P | MATCH_same selection predicate, census gain 0.15 |

## (a) Full 180-row table (sorted by causal importance)
L2 column shows the inherited per-layer driver number (marginal dCE ± SE); layer-0 rows show "whole-model §33".

| # | component | weight | L1 | L2 (layer dCE±SE) | L3 | L4 | L5 | anatomy / mechanism / name (with § refs) |
|---|---|---|---|---|---|---|---|---|
| 1 | mlp.L1 | 5.5744 | Y | Y (0.00052±0.00016) | Y | Y | P | THE HUB: five named stream-pair terms ARE the block (keep-5 = +0.0019 of a 5.574 floor, red-teamed, §86); interaction device, not additive; redundant distributed code — sufficiency without necessity (§83), compactly hierarchical (in-288 x out-144 keeps ~97%, §84/§85); feeds the two-branch induction MATCH fabric (§32b/§44); SAE names but does not explain it (§78-§81); L5 partial via the gated five-term sufficiency anatomy, no single named algorithm [floor 5.574±0.032, 3 terms to 95%: ArxMr+ArxAr+MrxMr] |
| 2 | mlp.L0 | 1.2341 | Y | Y* (whole-model §33) | Y | P | N | category-engine member (§32/§44); block-0 bilinear MLP opened in RESULTS §7; compact anatomy: 2 terms (ExE + ExAr) reach 95% (§89); no verified block-level algorithm [floor 1.234±0.013, 2 terms to 95%: ExE+ExAr] |
| 3 | mlp.L2 | 0.7390 | Y | Y (0.00136±0.00020) | Y | Y | P | "SQUARE THE PREVIOUS BLOCK'S OUTPUT": MrxMr alone leaves 0.093 of the 0.739 floor; two named terms = 98% of the layer (§89); category-engine member (§44) [floor 0.739±0.009, 2 terms to 95%: MrxMr+ArxMr] |
| 4 | mlp.L3 | 0.6163 | Y | Y (0.00093±0.00017) | Y | P | N | category-engine member (§32/§44); 4-term anatomy (§89); no block-level mechanism statement [floor 0.616±0.008, 4 terms to 95%: MrxMr+ArxMr+MexMr+ArxAr] |
| 5 | mlp.L17 | 0.4206 | Y | Y (0.00045±0.00012) | Y | Y | Y | DIFFERENTIAL PAIR readout: mlp-recent^2 writes a broad lexical-class prior, attention-earlier x mlp-recent writes its near-negation (class-signature cosine -0.965); the computation is the context-conditioned DIFFERENCE, sharpening at structural decision points (§91); d1 = the genuine context-conditioned CAPITAL SELECTOR (specificity 3.24, §69) with a calibrated placebo-controlled dial (§75) — the gate-passing named algorithm; d2/d3/d0 = static class priors (§69/§76); term knobs + input-side transplant editing law (§93/§93b/§94) [floor 0.421±0.009, 10 terms to 95%: AexMr+AexMe+MexMr+MrxMr...] |
| 6 | mlp.L4 | 0.1481 | Y | Y (0.00274±0.00027) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 5 terms to 95% (§89) [floor 0.148±0.004, 5 terms to 95%: ArxMr+MrxMr+ArxAr+MexMr...] |
| 7 | mlp.L16 | 0.1478 | Y | Y (0.00242±0.00021) | Y | Y | N | PURE HISTORY-READER: MexMe + AexMe leave 0.033 of the 0.148 floor, own attention causally dead (§89); lexical-readout family (§44); direction mechanisms verified: d0 WORD-class suppressor (§68), d1 newline->capital booster (no specificity control possible, §64/§66), d2 WORD-class pusher = static frequency prior (ratio 0.18, anti-selective — failed the §69 selector gate) [floor 0.148±0.005, 6 terms to 95%: MexMe+AexMe+AexMr+MexMr...] |
| 8 | mlp.L5 | 0.0928 | Y | Y (0.00242±0.00026) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 8 terms to 95% (§89) [floor 0.093±0.003, 8 terms to 95%: ArxMr+ArxMe+ArxAr+MexMr...] |
| 9 | mlp.L6 | 0.0832 | Y | Y (0.00313±0.00029) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 10 terms to 95% (§89) [floor 0.083±0.003, 10 terms to 95%: AexMr+MrxMr+MexMr+ArxMr...] |
| 10 | h.L0.3 | 0.0725 | Y | Y* (whole-model §33) | Y | Y | Y | prev-token head (FIXED-OFFSET back-1, purity 0.69, §62; causal +0.074±0.003) + verified CAPITAL class-pusher (specificity z 5.9, §68); largest single head path in the census (§67) |
| 11 | mlp.L7 | 0.0601 | Y | Y (0.00186±0.00025) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 12 terms to 95% (§89) [floor 0.060±0.003, 12 terms to 95%: MexMr+AexMe+AexMr+AexAr...] |
| 12 | mlp.L9 | 0.0563 | Y | Y (0.00109±0.00017) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 10 terms to 95% (§89); direction d1 is a PURE punctuation detector that is causally null — trigger-genuine/output-diffuse (§63) [floor 0.056±0.002, 10 terms to 95%: AexMe+MexMr+MexMe+ArxMe...] |
| 13 | mlp.L8 | 0.0511 | Y | Y (0.00313±0.00030) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 10 terms to 95% (§89) [floor 0.051±0.002, 10 terms to 95%: AexMe+MexMr+AexMr+MexMe...] |
| 14 | mlp.L11 | 0.0466 | Y | Y (0.00059±0.00016) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 10 terms to 95% (§89); §58 punctuation-position feature-builder bucket member [floor 0.047±0.002, 10 terms to 95%: AexMe+MexMe+MexMr+AexAe...] |
| 15 | mlp.L10 | 0.0465 | Y | Y (0.00081±0.00015) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 10 terms to 95% (§89) [floor 0.046±0.002, 10 terms to 95%: AexMe+MexMe+MexMr+AexMr...] |
| 16 | mlp.L12 | 0.0437 | Y | Y (0.00014±0.00010) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 6 terms to 95% (§89); §58 punctuation-position feature-builder bucket member [floor 0.044±0.002, 6 terms to 95%: AexMe+MexMe+MexMr+AexAe...] |
| 17 | mlp.L13 | 0.0396 | Y | Y (0.00171±0.00019) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 8 terms to 95% (§89) [floor 0.040±0.002, 8 terms to 95%: MexMe+AexMe+MexMr+AexAe...] |
| 18 | mlp.L15 | 0.0394 | Y | Y (0.00081±0.00012) | Y | P | N | direction d2 = verified punctuation->capital remap (drop 0.0068±0.0009, z 7.7, full control, §64) but §66 arc: output is a GENERIC shared capital direction (boundary-over-proper-noun specificity 1.0) — a boundary-triggered booster, not a sentence-boundary algorithm; d1 twin is a sign-inverted proxy artifact (§64); block otherwise mid-stack refinement (§44); 6-term anatomy (§89) [floor 0.039±0.002, 6 terms to 95%: MexMe+AexMe+MexMr+AexAe...] |
| 19 | mlp.L14 | 0.0300 | Y | Y (0.00380±0.00030) | Y | N | N | mid-stack distributed refinement — no distinct family (§44); term anatomy measured: 8 terms to 95% (§89) [floor 0.030±0.002, 8 terms to 95%: MexMe+AexMe+MexMr+AexAe...] |
| 20 | h.L1.1 | 0.0296 | Y | Y (0.00052±0.00016) | Y | P | N | self-attention head (FIXED-OFFSET back-0, purity 0.75); load-bearing +0.030±0.002, damage uniform across line structure (§62); function of the routed content unnamed |
| 21 | h.L6.3 | 0.0262 | Y | Y (0.00313±0.00029) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.43, §62); routed content/function unnamed |
| 22 | h.L7.8 | 0.0194 | Y | Y (0.00186±0.00025) | Y | P | P | verified distributed SUBWORD class-pusher (specificity z 6.4, §76); FIXED-OFFSET back-1 (§62); one of only 2 newly nameable of the top-30 unnamed (§76) |
| 23 | h.L9.7 | 0.0184 | Y | Y (0.00109±0.00017) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.57, §62); routed content/function unnamed |
| 24 | h.L7.0 | 0.0170 | Y | Y (0.00186±0.00025) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 25 | h.L4.0 | 0.0148 | Y | Y (0.00274±0.00027) | Y | Y | P | year/number head that QK-STEERS where h.L6.7 attends: direct-edge patch +0.278±0.056 (z 4.9), 33x specificity control (§57); edge gate passed, head-level output algorithm unnamed; offset-envelope positional (§62) |
| 26 | h.L4.1 | 0.0140 | Y | Y (0.00274±0.00027) | Y | P | N | FIXED-OFFSET back-1 (purity 0.40); structural head, class-diffuse output; upstream lesion RAISES its necessity 3.3x (§67); additive pair with h.L0.8 (§68) |
| 27 | h.L11.6 | 0.0128 | Y | Y (0.00059±0.00016) | P | N | N | weak positional signature (FIXED-OFFSET (back-1), purity 0.31, §62); otherwise census fingerprint only |
| 28 | h.L2.5 | 0.0122 | Y | Y (0.00136±0.00020) | Y | Y | Y | MATCH_same induction core (census gain 0.249); meaning-verified induction MATCH predicate, substitution-gated held-out (§47/§49); §62 label: FIXED-OFFSET (back-0) (purity 0.68) |
| 29 | h.L4.5 | 0.0118 | Y | Y (0.00274±0.00027) | P | N | N | weak positional signature (FIXED-OFFSET (back-1), purity 0.40, §62); otherwise census fingerprint only |
| 30 | h.L5.7 | 0.0114 | Y | Y (0.00242±0.00026) | Y | P | N | absolute-position-0 sink (purity 0.96); causal +0.0114±0.0012; shows an early monotone distance rise the §62 red-team flagged (saturating-signal caveat); §62 label: ABS-POS-0 SINK (purity 0.96) |
| 31 | h.L1.4 | 0.0109 | Y | Y (0.00052±0.00016) | Y | P | P | MATCH_same selection predicate, census gain 0.15; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not; also FIXED-OFFSET back-0 (§62) |
| 32 | h.L5.6 | 0.0107 | Y | Y (0.00242±0.00026) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.39, §62); otherwise census fingerprint only |
| 33 | h.L11.2 | 0.0106 | Y | Y (0.00059±0.00016) | Y | Y | P | verified WORD-class SUPPRESSOR (ablation raises word logits, z -4.6; §68 — corrects the §67 completion-predictor guess); PREV1 census label (0.081); diffuse trigger, distributed output; §62 label: FIXED-OFFSET (back-1) (purity 0.42) |
| 34 | h.L1.3 | 0.0099 | Y | Y (0.00052±0.00016) | Y | P | P | PREV1 selection predicate, census gain 0.17; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not; also FIXED-OFFSET back-1 (§62) |
| 35 | h.L2.6 | 0.0098 | Y | Y (0.00136±0.00020) | Y | P | N | FIXED-OFFSET (back-1) positional head (purity 0.79, §62); routed content/function unnamed |
| 36 | h.L5.8 | 0.0095 | Y | Y (0.00242±0.00026) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 37 | h.L4.7 | 0.0086 | Y | Y (0.00274±0.00027) | Y | P | N | FIXED-OFFSET (back-1) positional head (purity 0.67, §62); routed content/function unnamed |
| 38 | h.L6.7 | 0.0086 | Y | Y (0.00313±0.00029) | Y | P | N | boundary head; verified TARGET of the L4.0 and mlp.L1 edges (§57); §60 false-positive control: looks successor-copy by proxy but ablation shows it is causal (dCE +0.115) NOT via copying — own algorithm unresolved |
| 39 | h.L3.4 | 0.0083 | Y | Y (0.00093±0.00017) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.42, §62); routed content/function unnamed |
| 40 | h.L2.3 | 0.0081 | Y | Y (0.00136±0.00020) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 41 | h.L3.6 | 0.0081 | Y | Y (0.00093±0.00017) | Y | P | N | FIXED-OFFSET (back-1) positional head (purity 0.64, §62); routed content/function unnamed |
| 42 | h.L5.3 | 0.0072 | Y | Y (0.00242±0.00026) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 43 | h.L1.5 | 0.0070 | Y | Y (0.00052±0.00016) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.43, §62); routed content/function unnamed |
| 44 | h.L6.1 | 0.0068 | Y | Y (0.00313±0.00029) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 45 | h.L2.2 | 0.0064 | Y | Y (0.00136±0.00020) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.45, §62); routed content/function unnamed |
| 46 | h.L10.5 | 0.0063 | Y | Y (0.00081±0.00015) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.38, §62); otherwise census fingerprint only |
| 47 | h.L5.5 | 0.0061 | Y | Y (0.00242±0.00026) | Y | Y | P | atlas induction head; MATCH_prev census (0.138); verbatim-copy member of the DISTRIBUTED copy circuit, in the minimal 4-head subset (§60/§61 joint gate, redundancy 3.86) |
| 48 | h.L14.4 | 0.0060 | Y | Y (0.00380±0.00030) | P | P | N | WORD-class prior at boundaries (§76 concrete example: class push z 13.1, ablation z 1.4 — below load-bearing bar); type = static class prior |
| 49 | h.L2.7 | 0.0059 | Y | Y (0.00136±0.00020) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 50 | h.L8.1 | 0.0056 | Y | Y (0.00313±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 51 | h.L0.6 | 0.0055 | Y | Y* (whole-model §33) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 52 | h.L3.5 | 0.0053 | Y | Y (0.00093±0.00017) | Y | P | P | MATCH_same selection predicate, census gain 0.13; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not; also FIXED-OFFSET back-0 (§62) |
| 53 | h.L0.8 | 0.0053 | Y | Y* (whole-model §33) | P | P | N | KEY_newline census label = §54 artifact (ordinary distributed capital/punct support); structural head with class-diffuse output (§67); additive cross-layer pair with h.L4.1 (§68) |
| 54 | h.L3.0 | 0.0051 | Y | Y (0.00093±0.00017) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 55 | h.L2.8 | 0.0049 | Y | Y (0.00136±0.00020) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.44, §62); routed content/function unnamed |
| 56 | h.L13.0 | 0.0049 | Y | Y (0.00171±0.00019) | Y | Y | P | copy head (place/country adjectives), strongest source-specificity (src-rand +1.078±0.054); real copy operation, not pivotal solo (§60) |
| 57 | h.L3.3 | 0.0049 | Y | Y (0.00093±0.00017) | Y | P | P | coordination/list-continuation head (and/or/comma -> next enumeration marker); verified but modest (z~2.8, CE +0.025, §56); FIXED-OFFSET back-0 label (§62) |
| 58 | h.L3.8 | 0.0048 | Y | Y (0.00093±0.00017) | Y | Y | Y | MATCH_same, strongest census head (gain 0.314), steer-confirmed; meaning-verified MATCH predicate gate (§47/§49); §62 label: FIXED-OFFSET (back-0) (purity 0.74) |
| 59 | h.L8.3 | 0.0048 | Y | Y (0.00313±0.00030) | Y | Y | Y | DIGIT COPY / value-router: damage on copyable next-digit positions only (+0.155 vs +0.000), boosts attended source (§65); digit orthographic trigger, out-of-sample purity 0.97, position-matched 7.6x (§63); §60 copy rank 1 |
| 60 | h.L8.7 | 0.0046 | Y | Y (0.00313±0.00030) | Y | Y | Y | source-INDEPENDENT next-number predictor: damage on NON-copyable digit positions, boosts correct next digit not the source (§65); digit trigger purity 0.90, position-matched ~4x (§63); also capital-class suppressor (§68) |
| 61 | h.L3.7 | 0.0044 | Y | Y (0.00093±0.00017) | Y | P | N | FIXED-OFFSET (back-1) positional head (purity 0.64, §62); routed content/function unnamed |
| 62 | h.L1.8 | 0.0041 | Y | Y (0.00052±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 63 | h.L9.8 | 0.0038 | Y | Y (0.00109±0.00017) | P | P | N | KEY_newline census predicate (gain 0.08) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 64 | h.L4.6 | 0.0038 | Y | Y (0.00274±0.00027) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 65 | h.L5.2 | 0.0037 | Y | Y (0.00242±0.00026) | Y | P | N | ABS-POS-0 SINK positional head (purity 0.67, §62); routed content/function unnamed |
| 66 | h.L9.6 | 0.0036 | Y | Y (0.00109±0.00017) | Y | Y | Y | sentence-boundary subject-pronoun predictor (boosts they/it/They/that after "."); causally verified +0.077±0.023, 3.3 SE (§56) |
| 67 | h.L5.1 | 0.0036 | Y | Y (0.00242±0.00026) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 68 | h.L1.7 | 0.0036 | Y | Y (0.00052±0.00016) | P | P | N | KEY_newline census predicate (gain 0.05) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 69 | h.L8.8 | 0.0035 | Y | Y (0.00313±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 70 | h.L0.7 | 0.0032 | Y | Y* (whole-model §33) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.19, §62); otherwise census fingerprint only |
| 71 | h.L2.0 | 0.0031 | Y | Y (0.00136±0.00020) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 72 | h.L9.1 | 0.0030 | Y | Y (0.00109±0.00017) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 73 | h.L13.8 | 0.0030 | Y | Y (0.00171±0.00019) | Y | Y | Y | v1-router: routes layer-0 value payloads; bracket-closer (§41) + quote-style (§50) with value-swap causal gates; punctuation orthographic trigger purity 1.00 in+out of sample (§63); re-derived unsupervised (§56) |
| 74 | h.L14.6 | 0.0028 | Y | Y (0.00380±0.00030) | Y | P | P | MATCH_prev selection predicate, census gain 0.06; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 75 | h.L5.4 | 0.0028 | Y | Y (0.00242±0.00026) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 76 | h.L10.4 | 0.0028 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 77 | h.L6.5 | 0.0026 | Y | Y (0.00313±0.00029) | Y | P | P | MATCH_prev selection predicate, census gain 0.12; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 78 | h.L4.3 | 0.0026 | Y | Y (0.00274±0.00027) | Y | N | N | FIXED-OFFSET (back-1) positional head (purity 0.47, §62); routed content/function unnamed |
| 79 | h.L17.6 | 0.0025 | Y | Y (0.00045±0.00012) | Y | P | N | degree-adverb suppressor (completely/even/just rise on ablation, z 19.7) but CE-neutral (§59); FIXED-OFFSET back-1 (§62) |
| 80 | h.L15.1 | 0.0025 | Y | Y (0.00081±0.00012) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.39, §62); otherwise census fingerprint only |
| 81 | h.L7.7 | 0.0025 | Y | Y (0.00186±0.00025) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 82 | h.L9.4 | 0.0024 | Y | Y (0.00109±0.00017) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 83 | h.L7.1 | 0.0023 | Y | Y (0.00186±0.00025) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 84 | h.L11.1 | 0.0023 | Y | Y (0.00059±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 85 | h.L0.0 | 0.0022 | Y | Y* (whole-model §33) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 86 | h.L11.5 | 0.0022 | Y | Y (0.00059±0.00016) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.34, §62); otherwise census fingerprint only |
| 87 | h.L4.4 | 0.0021 | Y | Y (0.00274±0.00027) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 88 | h.L10.2 | 0.0020 | Y | Y (0.00081±0.00015) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.41, §62); routed content/function unnamed |
| 89 | h.L7.2 | 0.0019 | Y | Y (0.00186±0.00025) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.14, §62); otherwise census fingerprint only |
| 90 | h.L9.3 | 0.0019 | Y | Y (0.00109±0.00017) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 91 | h.L6.6 | 0.0019 | Y | Y (0.00313±0.00029) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.39, §62); otherwise census fingerprint only |
| 92 | h.L11.3 | 0.0019 | Y | Y (0.00059±0.00016) | Y | Y | P | subject-verb agreement router: reads the head-noun position (weight-share 0.35 vs 0.05 attractor), consumes an EARLY residual number feature (§42/§53, identity controls passed); redundant (ablation keeps accuracy 1.00); also a subword-continuation class prior (§76); §62 label: FIXED-OFFSET (back-0) (purity 0.42) |
| 93 | h.L1.0 | 0.0018 | Y | Y (0.00052±0.00016) | P | N | N | structure/newline cluster member (§58); §61 joint-ablation verdict: GENUINELY UNIMPORTANT (joint ~= sum of solos, ratio 1.12, fails random control) — not redundant, just null |
| 94 | h.L5.0 | 0.0018 | Y | Y (0.00242±0.00026) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 95 | h.L13.5 | 0.0018 | Y | Y (0.00171±0.00019) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 96 | h.L16.3 | 0.0018 | Y | Y (0.00242±0.00021) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 97 | h.L4.8 | 0.0017 | Y | Y (0.00274±0.00027) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.25, §62); otherwise census fingerprint only |
| 98 | h.L0.5 | 0.0016 | Y | Y* (whole-model §33) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 99 | h.L11.4 | 0.0016 | Y | Y (0.00059±0.00016) | P | P | N | KEY_newline census predicate (gain 0.09) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 100 | h.L8.2 | 0.0015 | Y | Y (0.00313±0.00030) | Y | P | N | line-boundary predictor (attends prev newline -> predicts newline); weak causal support (1.9 SE, §56); #1 cleanliness path yet not statistically load-bearing (§67) |
| 101 | h.L1.6 | 0.0015 | Y | Y (0.00052±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 102 | h.L17.0 | 0.0015 | Y | Y (0.00045±0.00012) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.30, §62); otherwise census fingerprint only |
| 103 | h.L2.1 | 0.0015 | Y | Y (0.00136±0.00020) | Y | P | P | MATCH_prev selection predicate, census gain 0.06; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not; §62: table-delimiter-ish, near-null solo (~0.001); §61 diffuse-cluster member |
| 104 | h.L6.0 | 0.0015 | Y | Y (0.00313±0.00029) | Y | Y | Y | sentence-boundary -> CAPITALIZED discourse-opener booster (Lastly/Finally/...); causal alt-control z=-12.25 (§56; causal step corrected the lowercase proxy read) |
| 105 | h.L14.7 | 0.0015 | Y | Y (0.00380±0.00030) | Y | Y | P | verbatim-copy head, source-specific drop verified, dCE~0 solo (§60) |
| 106 | h.L0.4 | 0.0014 | Y | Y* (whole-model §33) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 107 | h.L8.4 | 0.0014 | Y | Y (0.00313±0.00030) | Y | Y | P | verbatim-copy head (year tokens); source-specific logit drop verified, dCE~0 solo (buffered); minimal copy-subset member (§60/§61) |
| 108 | h.L15.3 | 0.0013 | Y | Y (0.00081±0.00012) | P | P | N | KEY_cap census cluster (gain 0.09); §46: capital-vs-lowercase is a STATIC PRIOR (survives cluster ablation at 101-102%) — the name FAILED the meaning gate; real +0.05-nat within-capital discrimination remains un-named |
| 109 | h.L11.0 | 0.0013 | Y | Y (0.00059±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 110 | h.L4.2 | 0.0013 | Y | Y (0.00274±0.00027) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 111 | h.L14.3 | 0.0013 | Y | Y (0.00380±0.00030) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.51, §62); routed content/function unnamed |
| 112 | h.L10.3 | 0.0013 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 113 | h.L8.6 | 0.0013 | Y | Y (0.00313±0.00030) | Y | P | P | MATCH_prev selection predicate, census gain 0.05; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 114 | h.L9.0 | 0.0012 | Y | Y (0.00109±0.00017) | P | N | N | weak positional signature (FIXED-OFFSET (back-1), purity 0.32, §62); otherwise census fingerprint only |
| 115 | h.L12.6 | 0.0012 | Y | Y (0.00014±0.00010) | Y | P | P | MATCH_prev selection predicate, census gain 0.12; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 116 | h.L17.2 | 0.0012 | Y | Y (0.00045±0.00012) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.33, §62); otherwise census fingerprint only |
| 117 | h.L12.4 | 0.0012 | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 118 | h.L1.2 | 0.0011 | Y | Y (0.00052±0.00016) | P | N | N | structure/newline cluster member (§58); §61 joint-ablation verdict: GENUINELY UNIMPORTANT (joint ~= sum of solos, ratio 1.12, fails random control) — not redundant, just null |
| 119 | h.L9.2 | 0.0011 | Y | Y (0.00109±0.00017) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.39, §62); otherwise census fingerprint only |
| 120 | h.L10.1 | 0.0011 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 121 | h.L16.0 | 0.0011 | Y | Y (0.00242±0.00021) | P | P | N | KEY_cap census cluster (gain 0.11); §46: capital-vs-lowercase is a STATIC PRIOR (survives cluster ablation at 101-102%) — the name FAILED the meaning gate; real +0.05-nat within-capital discrimination remains un-named |
| 122 | h.L6.8 | 0.0011 | Y | Y (0.00313±0.00029) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 123 | h.L11.8 | 0.0011 | Y | Y (0.00059±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 124 | h.L17.1 | 0.0011 | Y | Y (0.00045±0.00012) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.33, §62); otherwise census fingerprint only |
| 125 | h.L16.4 | 0.0011 | Y | Y (0.00242±0.00021) | P | P | N | KEY_newline census predicate (gain 0.14) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 126 | h.L7.4 | 0.0011 | Y | Y (0.00186±0.00025) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 127 | h.L11.7 | 0.0011 | Y | Y (0.00059±0.00016) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 128 | h.L7.6 | 0.0010 | Y | Y (0.00186±0.00025) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 129 | h.L14.1 | 0.0010 | Y | Y (0.00380±0.00030) | Y | P | P | MATCH_same selection predicate, census gain 0.07; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 130 | h.L3.1 | 0.0010 | Y | Y (0.00093±0.00017) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 131 | h.L14.8 | 0.0010 | Y | Y (0.00380±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 132 | h.L10.6 | 0.0010 | Y | Y (0.00081±0.00015) | P | P | N | KEY_newline census predicate (gain 0.06) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 133 | h.L13.6 | 0.0009 | Y | Y (0.00171±0.00019) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.47, §62); routed content/function unnamed |
| 134 | h.L14.0 | 0.0008 | Y | Y (0.00380±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 135 | h.L12.3 | 0.0008 | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 136 | h.L10.0 | 0.0008 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 137 | h.L7.5 | 0.0008 | Y | Y (0.00186±0.00025) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.36, §62); otherwise census fingerprint only |
| 138 | h.L17.8 | 0.0007 | Y | Y (0.00045±0.00012) | P | N | N | weak positional signature (POSITIONAL (offset-envelope), purity 0.29, §62); otherwise census fingerprint only |
| 139 | h.L12.1 | 0.0006 | Y | Y (0.00014±0.00010) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.36, §62); otherwise census fingerprint only |
| 140 | h.L17.4 | 0.0006 | Y | Y (0.00045±0.00012) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.52, §62); routed content/function unnamed |
| 141 | h.L0.1 | 0.0006 | Y | Y* (whole-model §33) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 142 | h.L9.5 | 0.0006 | Y | Y (0.00109±0.00017) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.41, §62); routed content/function unnamed |
| 143 | h.L12.8 | 0.0006 | Y | Y (0.00014±0.00010) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.34, §62); otherwise census fingerprint only |
| 144 | h.L13.4 | 0.0006 | Y | Y (0.00171±0.00019) | Y | P | P | verified distributed SUBWORD class-pusher (specificity z 3.5, §76) |
| 145 | h.L12.0 | 0.0005 | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 146 | h.L16.8 | 0.0005 | Y | Y (0.00242±0.00021) | Y | N | N | FIXED-OFFSET (back-0) positional head (purity 0.56, §62); routed content/function unnamed |
| 147 | h.L17.3 | 0.0005 | Y | Y (0.00045±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 148 | h.L7.3 | 0.0005 | Y | Y (0.00186±0.00025) | Y | Y | P | MATCH_prev census (0.107); copy head (dates/months), minimal copy-subset member (§60/§61) |
| 149 | h.L13.3 | 0.0004 | Y | Y (0.00171±0.00019) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 150 | h.L15.4 | 0.0004 | Y | Y (0.00081±0.00012) | P | P | N | KEY_cap census cluster (gain 0.09); §46: capital-vs-lowercase is a STATIC PRIOR (survives cluster ablation at 101-102%) — the name FAILED the meaning gate; real +0.05-nat within-capital discrimination remains un-named |
| 151 | h.L10.8 | 0.0004 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 152 | h.L15.7 | 0.0004 | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 153 | h.L15.6 | 0.0004 | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 154 | h.L16.1 | 0.0004 | Y | Y (0.00242±0.00021) | P | P | N | KEY_cap census cluster (gain 0.07); §46: capital-vs-lowercase is a STATIC PRIOR (survives cluster ablation at 101-102%) — the name FAILED the meaning gate; real +0.05-nat within-capital discrimination remains un-named |
| 155 | h.L10.7 | 0.0004 | Y | Y (0.00081±0.00015) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 156 | h.L6.4 | 0.0003 | Y | Y (0.00313±0.00029) | P | N | N | weak positional signature (FIXED-OFFSET (back-0), purity 0.37, §62); otherwise census fingerprint only |
| 157 | h.L16.7 | 0.0003 | Y | Y (0.00242±0.00021) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 158 | h.L12.2 | 0.0003 | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 159 | h.L3.2 | 0.0003 | Y | Y (0.00093±0.00017) | P | P | N | KEY_newline census predicate (gain 0.05) = §54 MEASUREMENT ARTIFACT — low pattern-R2, inconsistent newline sign, newline causally inert; really an ordinary distributed capital/punctuation-supporting head |
| 160 | h.L13.2 | 0.0003 | Y | Y (0.00171±0.00019) | Y | P | P | MATCH_prev selection predicate, census gain 0.12; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 161 | h.L0.2 | 0.0003 | Y | Y* (whole-model §33) | P | N | N | structure/newline cluster member (§58); §61 joint-ablation verdict: GENUINELY UNIMPORTANT (joint ~= sum of solos, ratio 1.12, fails random control) — not redundant, just null |
| 162 | h.L12.5 | 0.0003 | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 163 | h.L16.5 | 0.0003 | Y | Y (0.00242±0.00021) | P | P | N | KEY_cap census cluster (gain 0.10); §46: capital-vs-lowercase is a STATIC PRIOR (survives cluster ablation at 101-102%) — the name FAILED the meaning gate; real +0.05-nat within-capital discrimination remains un-named |
| 164 | h.L6.2 | 0.0003 | Y | Y (0.00313±0.00029) | Y | P | N | KEY_func selection predicate, census gain 0.06; predicate gated in the §49 selection sweep (copy/induction/match family) — selection side named, content/output side not |
| 165 | h.L13.7 | 0.0003 | Y | Y (0.00171±0.00019) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 166 | h.L15.2 | 0.0002 | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 167 | h.L13.1 | 0.0002 | Y | Y (0.00171±0.00019) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 168 | h.L17.7 | 0.0001 | Y | Y (0.00045±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 169 | h.L15.0 | 0.0001 | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 170 | h.L16.6 | 0.0000 | Y | Y (0.00242±0.00021) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 171 | h.L8.0 | 0.0000 | Y | Y (0.00313±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 172 | h.L2.4 | 0.0000 (raw -0.0003) | Y | Y (0.00136±0.00020) | Y | P | N | LINE-STRUCTURE head (attends last newline, 42% of queries) but causally NULL solo (-0.0003, §62); KEY_newline census label is a §54 artifact |
| 173 | h.L8.5 | 0.0000 (raw -0.0001) | Y | Y (0.00313±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 174 | h.L12.7 | 0.0000 (raw -0.0003) | Y | Y (0.00014±0.00010) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 175 | h.L14.2 | 0.0000 (raw -0.0005) | Y | Y (0.00380±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 176 | h.L14.5 | 0.0000 (raw -0.0000) | Y | Y (0.00380±0.00030) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 177 | h.L15.5 | 0.0000 (raw -0.0000) | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 178 | h.L15.8 | 0.0000 (raw -0.0000) | Y | Y (0.00081±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |
| 179 | h.L16.2 | 0.0000 (raw -0.0002) | Y | Y (0.00242±0.00021) | P | N | N | the cleanest census path overall, yet ablating it slightly HELPS (dCE z=-4.1) — the flagship cleanliness!=importance case (§67) |
| 180 | h.L17.5 | 0.0000 (raw -0.0000) | Y | Y (0.00045±0.00012) | N | N | N | census fingerprint only (trigger/effect measured, §67); no verified type-detector characterization |

---
Compact legend: L1 represented (exact rewrite) · L2 substitutable (per-layer driver) · L3 anatomy (terms/detector/predicate)
· L4 mechanism (HOW statement) · L5 named-algorithm (gate-passed). Y/P/N. Weights: solo mean-ablation dCE (head path /
MLP-block floor), held-back FW[448:600]. Assembled from existing artifacts; do not read the mass fractions as fractions
of joint headroom (see §71).
