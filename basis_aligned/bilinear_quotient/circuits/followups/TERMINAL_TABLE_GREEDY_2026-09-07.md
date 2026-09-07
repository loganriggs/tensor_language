# Terminal evidence — removal-greedy head sets (generated 2026-09-07 00:37 UTC by ops/terminal_table_greedy_sets.py)

Rows evaluated on the ODD half of each family (directions = per-block diff-in-means fit on the EVEN half); rubric rows: 2 extraction ≥0.80 (LB ≥0.60), 3 removal LB>0 with own-C specificity LB>0, 4 own-C CE UB ≤0.01, 5 A1-fit direction on A2 LB>0 and ≥0.50× A1. Receipts: v66/v67 (curves, cross), v68 (extraction), v69 (rows 3–5), v70 (C-penalised).

| behaviour | set | n | removal A1 (LB) | extraction (LB) | own C (UB) | A2 a1-fit | A2 a2-fit | random | max cross | rows 2/3/4/5 |
|---|---|---|---|---|---|---|---|---|---|---|
| quantifier_number | hub | 2 | 0.353 | 0.509 (0.485) | — | — | — | — | — | ✗/·/·/· |
| quantifier_number | hub+3 | 5 | 0.539 (0.442) | 0.665 (0.634) | — | — | — | — | — | ✗/·/·/· |
| quantifier_number | hub+8 | 10 | 0.720 (0.572) | 0.816 (0.775) | -0.059 (-0.035) | 0.716 | 0.700 | -0.001 | -0.010 | ✓/✓/✓/✓ |
| verb_preposition | hub | 3 | 0.202 | 0.588 (0.566) | — | — | — | — | — | ✗/·/·/· |
| verb_preposition | hub+3 | 6 | 0.476 (0.397) | 0.840 (0.803) | — | — | — | — | — | ✓/·/·/· |
| verb_preposition | hub+8 | 11 | 0.723 (0.618) | 0.995 (0.950) | -0.241 (-0.212) | 0.542 | 0.692 | 0.003 | 0.020 | ✓/✓/✓/✓ |
| polarity_licensing | hub | 4 | 0.189 | 0.598 (0.574) | — | — | — | — | — | ✗/·/·/· |
| polarity_licensing | hub+3 | 7 | 0.386 (0.203) | 0.805 (0.780) | — | — | — | — | — | ✓/·/·/· |
| polarity_licensing | hub+8 | 12 | 0.567 (0.349) | 0.903 (0.873) | 0.016 (0.041) | 0.634 | 0.761 | -0.000 | 0.043 | ✓/✓/✗/✓ |
| polarity_licensing | hub+8 C-pen (λ=2) | 12 | 0.551 (0.339) | 0.908 (0.877) | 0.006 (0.030) | 0.585 | — | — | — | ✓/✓/✗/✓ |
| dative | hub | 5 | 0.282 | 0.638 (0.589) | — | — | — | — | — | ✗/·/·/· |
| dative | hub+3 | 8 | 0.388 (0.311) | 0.776 (0.715) | — | — | — | — | — | ✗/·/·/· |
| dative | hub+8 | 13 | 0.512 (0.411) | 0.877 (0.827) | -0.159 (-0.136) | 0.218 | 0.629 | -0.003 | 0.069 | ✓/✓/✓/✗ |
| verb_complementizer | hub | 3 | 0.598 | 0.591 (0.565) | — | — | — | — | — | ✗/·/·/· |
| verb_complementizer | hub+3 | 6 | 0.900 (0.688) | 0.766 (0.740) | — | — | — | — | — | ✗/·/·/· |
| verb_complementizer | hub+8 | 11 | 1.119 (0.863) | 0.893 (0.869) | 0.387 (0.483) | 0.679 | 0.820 | 0.004 | 0.008 | ✓/✓/✗/✓ |
| verb_complementizer | hub+8 C-pen (λ=2) | 11 | 1.034 (0.757) | 0.824 (0.794) | 0.284 (0.355) | 0.567 | — | — | — | ✓/✓/✗/✓ |
| voice_frame | hub | 4 | 0.092 | 0.606 (0.548) | — | — | — | — | — | ✗/·/·/· |
| voice_frame | hub+3 | 7 | 0.180 (0.125) | 0.708 (0.648) | — | — | — | — | — | ✗/·/·/· |
| voice_frame | hub+8 | 12 | 0.322 (0.190) | 0.809 (0.750) | -0.001 (0.030) | 0.610 | 0.609 | 0.004 | 0.028 | ✓/✓/✗/✓ |
| voice_frame | hub+8 C-pen (λ=2) | 12 | 0.276 (0.156) | 0.756 (0.696) | -0.119 (-0.093) | 0.635 | — | — | — | ✗/✓/✓/✓ |

## Sets

- **quantifier_number** hub ['attn:07:head:08', 'attn:11:head:03'] → additions in order ['05:head:03', '08:head:01', '13:head:01', '09:head:07', '02:head:02', '09:head:01', '04:head:01', '13:head:06']
- **verb_preposition** hub ['attn:06:head:03', 'attn:13:head:08', 'attn:08:head:08'] → additions in order ['11:head:03', '14:head:08', '07:head:08', '14:head:03', '08:head:01', '16:head:08', '10:head:08', '13:head:03']
- **polarity_licensing** hub ['attn:07:head:08', 'attn:08:head:01', 'attn:04:head:07', 'attn:03:head:00'] → additions in order ['09:head:07', '13:head:01', '10:head:05', '05:head:08', '16:head:08', '04:head:01', '15:head:01', '05:head:03']; C-penalised additions ['09:head:07', '13:head:01', '10:head:05', '05:head:08', '16:head:08', '15:head:01', '05:head:03', '09:head:03']
- **dative** hub ['attn:14:head:08', 'attn:07:head:08', 'attn:06:head:03', 'attn:13:head:08', 'attn:11:head:03'] → additions in order ['14:head:03', '08:head:01', '09:head:07', '05:head:08', '16:head:08', '04:head:01', '03:head:06', '11:head:02']
- **verb_complementizer** hub ['attn:06:head:03', 'attn:11:head:03', 'attn:07:head:08'] → additions in order ['08:head:01', '04:head:01', '09:head:03', '09:head:07', '14:head:03', '05:head:08', '09:head:01', '08:head:08']; C-penalised additions ['08:head:01', '04:head:01', '09:head:07', '09:head:03', '02:head:06', '01:head:05', '14:head:08', '05:head:07']
- **voice_frame** hub ['attn:07:head:08', 'attn:01:head:05', 'attn:00:head:03', 'attn:04:head:01'] → additions in order ['09:head:07', '16:head:08', '11:head:07', '10:head:05', '06:head:03', '02:head:02', '02:head:06', '02:head:03']; C-penalised additions ['16:head:08', '11:head:07', '10:head:05', '06:head:03', '02:head:02', '03:head:06', '02:head:06', '02:head:03']

## Notes

- verb_complementizer's C family ("The leader noted/replied quickly → that", foil whether) is the same that/whether prediction from that-taking verbs; the hub alone damages it by 0.31 (v70 curve k=0) on the that/whether margin (v69: margin −2.1, KL 0.03). Row 4 as written fails at every set size; the direction is a verb-class axis shared by all reporting verbs. v57's C for the verb sets was polarity's C (borrowed control) and is superseded.
- Row 4 on 16-document halves: bootstrap half-width ≈0.025, so UB ≤0.01 is unreachable at zero mean (polarity: point 0.006, UB 0.030). Read polarity's row 4 from full rows (v51) or as point+width.
- dative's A2 deficit is direction-keyed (A2-fit 0.63 vs A1-fit 0.22 at hub+8; v61/v62), unchanged by enlargement.
- Cross-collateral (A1-fit direction on the other five A1 families, odd rows): max 0.069 (dative→verb set), otherwise ≤0.043; quantifier's direction LOWERS dative/polarity CE by 0.11–0.13 (shared number axis, v54).
