# Terminal evidence — removal-greedy head sets (generated 2026-09-07 03:03 UTC by ops/terminal_table_greedy_sets.py)

Rows evaluated on the ODD half of each family (directions = per-block diff-in-means fit on the EVEN half); rubric rows: 2 extraction ≥0.80 (LB ≥0.60), 3 removal LB>0 with own-C specificity LB>0, 4 own-C CE UB ≤0.01, 5 A1-fit direction on A2 LB>0 and ≥0.50× A1. Receipts: v66/v67 (curves, cross), v68 (extraction), v69 (rows 3–5), v70 (C-penalised), v75 (constrained DAS: direction = rank-1-per-block DAS on pooled A1 + v15 verb-variant EVEN rows, complement term, C-removal-inertness regularizer λ=30 on C EVEN rows; A2 and the odd rows never fitted), v76 (extraction of those directions on A1 and A2 odd rows).

| behaviour | set | n | removal A1 (LB) | extraction (LB) | own C (UB) | A2 a1-fit | A2 a2-fit | random | max cross | rows 2/3/4/5 |
|---|---|---|---|---|---|---|---|---|---|---|
| quantifier_number | hub | 2 | 0.353 | 0.509 (0.485) | — | — | — | — | — | ✗/·/·/· |
| quantifier_number | hub+3 | 5 | 0.539 (0.442) | 0.665 (0.634) | — | — | — | — | — | ✗/·/·/· |
| quantifier_number | hub+8 | 10 | 0.720 (0.572) | 0.816 (0.775) | -0.059 (-0.035) | 0.716 | 0.700 | -0.001 | -0.010 | ✓/✓/✓/✓ |
| quantifier_number | hub+8 constrained DAS (pooled, λ=30) | 10 | 0.738 (0.538) | 0.753 (0.705); A2 0.853 | 0.009 (0.020); λ=0 -0.046 | 0.866 | — | -0.003 | 0.022 | ✗/✓/✗/✓ |
| quantifier_number | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 10 | 0.731 (0.533) | 0.757 (0.710); A2 0.850 | 0.009 (0.018) | 0.887 | unseen pair 0.677 | -0.003 | max abs 0.009 | ✗/✓/✗/✓ |
| quantifier_number | hub+16 (greedy continuation v87, dim) | 18 | 0.925 (0.709) | 0.915 (0.870); A2 0.911 | -0.062 (-0.036) | 0.819 | — | — | max abs 0.134 | ✓/✓/✓/✓ |
| quantifier_number | hub+16 full-specificity DAS (v88) | 18 | 0.777 (0.597) | 0.848 (0.796); A2 0.853 | -0.004 (0.003) | 0.815 | — | ext -0.005 | max abs 0.009 | ✓/✓/✓ (cross-fit 64 docs, C -0.003 UB 0.003)/✓ |
| verb_preposition | hub | 3 | 0.202 | 0.588 (0.566) | — | — | — | — | — | ✗/·/·/· |
| verb_preposition | hub+3 | 6 | 0.476 (0.397) | 0.840 (0.803) | — | — | — | — | — | ✓/·/·/· |
| verb_preposition | hub+8 | 11 | 0.723 (0.618) | 0.995 (0.950) | -0.241 (-0.212) | 0.542 | 0.692 | 0.003 | 0.020 | ✓/✓/✓/✓ |
| verb_preposition | hub+8 constrained DAS (pooled, λ=30) | 11 | 0.906 (0.849) | 1.067 (1.017); A2 0.952 | -0.001 (0.003); λ=0 -0.082 | 0.655 | — | -0.001 | 0.077 | ✓/✓/✓/✓ |
| verb_preposition | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 11 | 0.871 (0.815) | 1.064 (1.015); A2 0.956 | -0.000 (0.003) | 0.636 | unseen pair 0.260 | -0.001 | max abs 0.005 | ✓/✓/✓/✓ |
| polarity_licensing | hub | 4 | 0.189 | 0.598 (0.574) | — | — | — | — | — | ✗/·/·/· |
| polarity_licensing | hub+3 | 7 | 0.386 (0.203) | 0.805 (0.780) | — | — | — | — | — | ✓/·/·/· |
| polarity_licensing | hub+8 | 12 | 0.567 (0.349) | 0.903 (0.873) | 0.016 (0.041) | 0.634 | 0.761 | -0.000 | 0.043 | ✓/✓/✗/✓ |
| polarity_licensing | hub+8 C-pen (λ=2) | 12 | 0.551 (0.339) | 0.908 (0.877) | 0.006 (0.030) | 0.585 | — | — | — | ✓/✓/✗/✓ |
| polarity_licensing | hub+8 constrained DAS (pooled, λ=30) | 12 | 0.666 (0.600) | 0.937 (0.889); A2 0.863 | 0.004 (0.008); λ=0 0.206 | 0.638 | — | -0.001 | 0.006 | ✓/✓/✓/✓ |
| polarity_licensing | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 12 | 0.648 (0.584) | 0.935 (0.884); A2 0.845 | 0.002 (0.009) | 0.661 | unseen pair 0.642 | -0.001 | max abs 0.005 | ✓/✓/✓/✓ |
| dative | hub | 5 | 0.282 | 0.638 (0.589) | — | — | — | — | — | ✗/·/·/· |
| dative | hub+3 | 8 | 0.388 (0.311) | 0.776 (0.715) | — | — | — | — | — | ✗/·/·/· |
| dative | hub+8 | 13 | 0.512 (0.411) | 0.877 (0.827) | -0.159 (-0.136) | 0.218 | 0.629 | -0.003 | 0.069 | ✓/✓/✓/✗ |
| dative | hub+8 constrained DAS (pooled, λ=30) | 13 | 0.565 (0.482) | 0.825 (0.776); A2 0.463 | 0.006 (0.021); λ=0 0.125 | 0.313 | — | 0.002 | 0.042 | ✓/✓/✗/✓ |
| dative | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 13 | 0.569 (0.487) | 0.826 (0.776); A2 0.446 | 0.013 (0.026) | 0.323 | unseen pair 0.443 | 0.002 | max abs 0.017 | ✓/✓/✗/✓ |
| dative | hub+8 full-specificity DAS, A2 EVEN in the fit pool (v89; A2 no longer held-out, the unseen fourth map is) | 13 | 0.579 (0.500) | 0.831 (0.780); A2 0.695 (0.633) | 0.015 (0.025) | 0.536 (in-pool) | unseen pair 0.474 | ext 0.001 | max abs 0.021 | ✓/✓/✗/✓ |
| verb_complementizer | hub | 3 | 0.598 | 0.591 (0.565) | — | — | — | — | — | ✗/·/·/· |
| verb_complementizer | hub+3 | 6 | 0.900 (0.688) | 0.766 (0.740) | — | — | — | — | — | ✗/·/·/· |
| verb_complementizer | hub+8 | 11 | 1.119 (0.863) | 0.893 (0.869) | 0.387 (0.483) | 0.679 | 0.820 | 0.004 | 0.008 | ✓/✓/✗/✓ |
| verb_complementizer | hub+8 C-pen (λ=2) | 11 | 1.034 (0.757) | 0.824 (0.794) | 0.284 (0.355) | 0.567 | — | — | — | ✓/✓/✗/✓ |
| verb_complementizer | hub+8 constrained DAS (pooled, λ=30) | 11 | 1.025 (0.784) | 0.880 (0.842); A2 0.835 | 0.029 (0.046); λ=0 0.733 | 0.714 | — | 0.004 | 0.033 | ✓/✓/✗/✓ |
| verb_complementizer | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 11 | 1.018 (0.775) | 0.873 (0.835); A2 0.823 | 0.021 (0.037) | 0.709 | unseen pair 0.720 | 0.004 | max abs 0.009 | ✓/✓/✗/✓ |
| verb_complementizer | hub+16 (greedy continuation v90, dim) | 19 | 1.291 (0.989) | 0.951 (0.925); A2 0.915 | 0.393 (0.493) | 0.793 | — | — | max abs 0.040 | ✓/✓/✗/✓ |
| verb_complementizer | hub+16 full-specificity DAS (v91) | 19 | 1.171 (0.862) | 0.939 (0.899); A2 0.868 | 0.021 (0.039) | 0.811 | — | ext 0.008 | max abs 0.012 | ✓/✓/✗ (cross-fit 64 docs, C 0.019 UB 0.033)/✓ |
| voice_frame | hub | 4 | 0.092 | 0.606 (0.548) | — | — | — | — | — | ✗/·/·/· |
| voice_frame | hub+3 | 7 | 0.180 (0.125) | 0.708 (0.648) | — | — | — | — | — | ✗/·/·/· |
| voice_frame | hub+8 | 12 | 0.322 (0.190) | 0.809 (0.750) | -0.001 (0.030) | 0.610 | 0.609 | 0.004 | 0.028 | ✓/✓/✗/✓ |
| voice_frame | hub+8 C-pen (λ=2) | 12 | 0.276 (0.156) | 0.756 (0.696) | -0.119 (-0.093) | 0.635 | — | — | — | ✗/✓/✓/✓ |
| voice_frame | hub+8 constrained DAS (pooled, λ=30) | 12 | 0.361 (0.193) | 0.754 (0.686); A2 0.732 | 0.007 (0.013); λ=0 0.026 | 0.716 | — | 0.004 | 0.023 | ✗/✓/✗/✓ |
| voice_frame | hub+8 full-specificity DAS (own C + 5 other A1 as inertness controls, 30 each) | 12 | 0.358 (0.196) | 0.752 (0.682); A2 0.726 | 0.004 (0.010) | 0.672 | unseen pair — | 0.004 | max abs 0.011 | ✗/✓/✗/✓ |
| voice_frame | hub+16 (greedy continuation v83, dim) | 20 | 0.526 (0.292) | 0.857 (0.809); A2 0.849 | 0.037 (0.066) | 0.701 | — | — | max abs 0.023 | ✓/✓/✗/✓ |
| voice_frame | hub+16 full-specificity DAS (v92) | 20 | 0.792 (0.472) | 0.808 (0.744); A2 0.781 | 0.002 (0.009) | 0.903 | — | ext 0.008 | max abs 0.009 | ✓/✓/✓ (cross-fit 64 docs, C 0.000 UB 0.005)/✓ |

## Sets

- **quantifier_number** hub ['attn:07:head:08', 'attn:11:head:03'] → additions in order ['05:head:03', '08:head:01', '13:head:01', '09:head:07', '02:head:02', '09:head:01', '04:head:01', '13:head:06']
- **verb_preposition** hub ['attn:06:head:03', 'attn:13:head:08', 'attn:08:head:08'] → additions in order ['11:head:03', '14:head:08', '07:head:08', '14:head:03', '08:head:01', '16:head:08', '10:head:08', '13:head:03']
- **polarity_licensing** hub ['attn:07:head:08', 'attn:08:head:01', 'attn:04:head:07', 'attn:03:head:00'] → additions in order ['09:head:07', '13:head:01', '10:head:05', '05:head:08', '16:head:08', '04:head:01', '15:head:01', '05:head:03']; C-penalised additions ['09:head:07', '13:head:01', '10:head:05', '05:head:08', '16:head:08', '15:head:01', '05:head:03', '09:head:03']
- **dative** hub ['attn:14:head:08', 'attn:07:head:08', 'attn:06:head:03', 'attn:13:head:08', 'attn:11:head:03'] → additions in order ['14:head:03', '08:head:01', '09:head:07', '05:head:08', '16:head:08', '04:head:01', '03:head:06', '11:head:02']
- **verb_complementizer** hub ['attn:06:head:03', 'attn:11:head:03', 'attn:07:head:08'] → additions in order ['08:head:01', '04:head:01', '09:head:03', '09:head:07', '14:head:03', '05:head:08', '09:head:01', '08:head:08']; C-penalised additions ['08:head:01', '04:head:01', '09:head:07', '09:head:03', '02:head:06', '01:head:05', '14:head:08', '05:head:07']
- **voice_frame** hub ['attn:07:head:08', 'attn:01:head:05', 'attn:00:head:03', 'attn:04:head:01'] → additions in order ['09:head:07', '16:head:08', '11:head:07', '10:head:05', '06:head:03', '02:head:02', '02:head:06', '02:head:03']; C-penalised additions ['16:head:08', '11:head:07', '10:head:05', '06:head:03', '02:head:02', '03:head:06', '02:head:06', '02:head:03']

## Notes

- verb_complementizer direction is ONE-SIDED (v93/v94): removal damage on the whether side is 1.97 (A1 odd), 2.13 (asked), 1.91 (inquired); on the that side 0.38/0.41/0.28 for the three pool declaratives (remarked/insisted/said) and 0.02–0.23 for ten unseen declaratives (noted 0.027, replied 0.030 — the row-4 sibling). The rank-1 direction is the interrogative marker; 'that' is what the model says when it is absent, plus a small fitted-verb component. All complementizer removal numbers in this table are side-pooled (that + whether)/2; the that-only sibling's 0.019 residual is the default side's leak. An unseen-interrogative sibling (debated/checked) is untested: the model prefers whether on only 62% of those rows.
- Side split for all six hub+8 directions (v95, ODD A1, nat): quantifier was 1.30 / were 0.16 (8x), complementizer whether 1.62 / that 0.42 (3.8x), dative for 0.78 / to 0.36 (2.2x), polarity anything 0.80 / something 0.50 (1.6x), voice by 0.44 / the 0.28 (1.6x), preposition on 0.94 / to 0.81 (1.2x). Only the number (singular side) and interrogative directions are strongly one-sided; their pooled removal understates the marked side ~2x. Markedness priors (NPI anything, passive by) did not predict one-sidedness.
- verb_complementizer C: the hub alone damages own C ("The leader noted/replied quickly → that", foil whether) by 0.31 (v70 curve k=0) on the that/whether margin (v69: margin −2.1, KL 0.03). Row 4 as written fails at every set size. v71: the direction is NOT a verb-class axis — it transfers only 0.26–0.41× to three unseen verb pairs (per-pair refits 1.7–3.1× stronger, block |cos| 0.43–0.50); single-pair directions are pair-keyed. v72: a POOLED direction (three pairs, 48 docs) transfers to the unseen pair at 0.56× (pooled diff-in-means 0.619), 0.61× (DAS+inertness 0.679), 0.67× (DAS 0.740) of its refit 1.110 while keeping the fitted pairs at ≥0.86×: a shared rank-1 axis exists and the single-pair fits were noisy samples. Pooling RAISES own-C damage (0.45–0.73). The reading posted at 00:42 ('C shares the that/whether output axis, row 4 unmeasurable') was REFUTED by v75: the C-inertness regularizer drives own C to 0.029 (UB 0.046) while keeping A1 at 1.025 (0.92× of the unconstrained 1.112) — the C-damaging component was separable and the C control is sound. Row 4 by the UB ≤0.01 bar is still not met for this set (residual 0.03, LB 0.011). v57 tested the verb sets against polarity's C (borrowed control) and is superseded.
- Row 4 on 16-document halves: bootstrap half-width ≈0.025, so UB ≤0.01 is unreachable at zero mean (polarity: point 0.006, UB 0.030). Read polarity's row 4 from full rows (v51) or as point+width.
- dative's A2 deficit is direction-keyed (A2-fit 0.63 vs A1-fit 0.22 at hub+8; v61/v62), unchanged by enlargement. v73/v74: a DAS direction pooled over A1 + two verb variants with a C-removal-inertness regularizer (λ=30, C even rows; `g.fit_block_subspace_constrained`) meets row 5 on odd rows (A2 0.313 = 0.55× A1 0.565, LB 0.279), transfers to an unseen verb pair at 0.87× its refit, keeps cross-collateral ≤0.042, and holds own C at 0.006 (UB 0.021 — misses the row-4 UB bar by bootstrap width). Without the regularizer the pooled direction damages C by 0.125.
- Cross-collateral (A1-fit direction on the other five A1 families, odd rows): max 0.069 (dative→verb set), otherwise ≤0.043; quantifier's direction LOWERS dative/polarity CE by 0.11–0.13 (shared number axis, v54).
