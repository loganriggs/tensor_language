# Reflexive person I/you — definition-of-done scorecard

Opened 2026-09-18 03:05 UTC (Claude lane; the PERSON readout family). Rubric: `better_circuits.md` §1.

**Path now.** A subject pronoun's person (I / you) one clause earlier → readout set {8.1, 13.1, 10.5, 15.1} along `O_h^T(u_myself − u_yourself)` at the final query → myself/yourself. Five live atlas
lines (reflexive person, object control, their plural forms, possessive my/your) share this set or three of it; 8.1, the temporal
family's token-only cue reader, leads or co-leads — here it reads a person-marked pronoun token. Readers avoid person / number:
will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 8.1 0.71, 13.1 0.62, 10.5 0.58, 15.1 0.40 (next 5.7 0.30); top-4 set 46%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.46 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, three unused frames: 'After the {obj} fell, I/you clearly blamed', 'The {agent}s say that I/you rarely trust', 'By the {obj} I/you then reminded') | edit, frozen | fresh (v104) | fractions after 0.47 / by 0.40 / say 0.40 (pooled 0.42); positive 96/96; null max 0.02; capability 1.00 in all six cells | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v104) | moves 0.10 / 0.04 / 0.05 vs null 0.05 / 0.06 / 0.04 (gate bar 0.62) | passes |
| 4 | Additive | edit | fresh (v104) | singles 13.1 0.73 / 8.1 0.63 / 10.5 0.48 / 15.1 0.37; gap 0.055 ≤ bar 0.092 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v104) | zero 2.03; retention 1.21; random keep ≤ 0.02 | passes |
| 6 | Matched-count random four-head-set null | edit | fresh rows (v106) | set 2.27 (fraction 0.42) vs random max 0.10 (fraction 0.02); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 8): the set's own writes carry 85% (attn:13 −0.61, attn:15 −0.44, attn:08 −0.43, attn:10 −0.43 of −2.24); downstream net −0.33 (MLP 10 −0.23, MLP 11 −0.11; MLP 17 +0.30 counteracts); remainder 1% | response | fresh (v107) | | passes 5/5 — a direct readout |
| 8 | Natural FineWeb rows (an I / you token within 12 tokens, next token myself / yourself; other person pronouns excluded): congruent cells (I/myself, you/yourself; capability 1.00) removal 2.18 of 5.93 (37%), positive 32/32, null max 0.03, selective (will−would moves 0.14 vs null 0.04, inside the gate); the 11 counter-case rows (a quoted speaker) shift toward the text's reflexive (−0.23) as registered | edit | natural (v110) | frozen v20 bars held | passes 6/6 |
| 9 | Pile rows (out-of-corpus): congruent 2.07 of 6.29 (33%), positive 32/32, null max 0.03, selective; 4 counter-case rows −0.05 | edit | natural OOD (v111) | | passes 6/6 |
| 10 | Source fold (exact, closure 2e-6): the four coefficients come from the I / you token position — 8.1 0.98 (91% through the token-only block-0 value), 13.1 0.98 (95% token-only), 15.1 0.80 (72% token-only), 10.5 0.52 (29% token-only; 0.24 final, 0.24 other); pooled 0.89 from the pronoun, 0.82 token-only | fold | fresh (v112) | registered "pooled token-only ≤ 0.50" false (0.82) — the opposite of the pronoun-number and have/has sets | 4/5 — a token-only reader family: three of four heads copy the pronoun token's block-0 value; the open port is their attention pattern, as for 8.1 on the temporal line |
| 11 | Token-only generator (§3.7) for {8.1, 13.1, 15.1}: replacing the three slices by p_h(final, pronoun) × λ_h × v1_h(pronoun) with the native pattern retains 91% of their zeroed service; with a CONSTANT pattern per (head, cue) taken from the other two constructions (leave-one-out, a number not a fit) 94% pooled — after 0.90, by 1.12, say 0.83; pattern CV ≤ 0.26 within every (head, construction, cue) | edit | fresh (v113) | | passes 5/5 — the three heads are a two-entry lookup table on the pronoun token; their ports are closed. 10.5 (contextual, 52% pronoun) stays open |
| 12 | Writer fold of the state 10.5 (the contextual member) reads at the I / you position (exact, closure 2e-7): MLPs 45% (MLP 9 0.13, MLP 6 0.11, MLP 4 / 5 / 7 ≈ 0.05 each), heads 30% (6.1 0.08, 8.1 0.06, 5.1 0.03), embedding 25%; no writer above 13% | fold | fresh (v118) | registered "MLP ≥ 0.50" false (0.45); "8.1 ≤ 0.10", "embedding ≥ 0.10", "largest MLP is 8 or 9" held | 4/5 — 10.5 reads a diffuse mix of the pronoun's token identity and early processing; declared port (no single writer to fold) |
| 13 | Shared heads, separate directions: at the person set the pronoun-number direction O_h^T(u_they − u_he) (that family shares 15.1 and 10.5) removes −0.001 of the myself/yourself margin; own 2.27; null max 0.005 | edit | fresh (v135) | | passes 5/5 |
| U1 | MLP 8 at UNIT grain on head 10.5's myself − yourself reader direction (v104 fresh rows, pooled I − you; closure 0.0): at the PRONOUN — top-5 1%, top-10 26%, top-50 84% (cancelling), min pairwise Jaccard of the three frames' top-50 0.09, leaders 3643 (23%), 198 (21%), 3986 (−18%); at the FINAL — top-5 47%, top-10 41%, top-50 70%, Jaccard 0.19, leaders 4404 (24%), 198 (18%), 2941 (−12%) | fold | v104 rows (v259) | closure held; "top-10 ≥ 0.50" false at both positions; "stable across frames ≥ 1/3" false | 1/4 — the person family has NO MLP-8 detector at the pronoun: MLP 8's part of 10.5's state is spread, mixed-sign and frame-specific — a PORT, as for the temporal family (v251–v253). The open member 10.5 of `in_depth_circuit.md` stays open at MLP-8 grain; function-word cues (I / you, since / by) are carried by token copies, not by MLP-8 units |
| U2 | MLP 7 at unit grain on 10.5's myself − yourself direction (v104 rows; closure 0.0), the one-block-lower check after v259: at the PRONOUN — top-5 15%, top-10 16%, top-50 13%, min frame Jaccard 0.18, leader 3200 (-20%); at the FINAL — top-5 42%, top-10 38%, top-50 63%, Jaccard 0.12 | fold | v104 rows (v284) | closure held; top-10 ≥ 0.50 false at both positions; stability false | 1/4 — spread at MLP 7 as at MLP 8 (v259): the person family is a PORT at both MLP grains; its state is the pronoun token carried by head copies (8.1 / 13.1 / 15.1) and no MLP unit names it. Declared |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on fresh, natural FineWeb and Pile rows (rows 2, 8, 9); the set follows the cue token (counter-cases move toward the text, as the gender set did and the number sets did not) | — |
| Extracted | held at the head boundary (row 5); direct (row 7); 8.1 / 13.1 / 15.1 close to a token-only generator with constant patterns (rows 10–11); 10.5's source is diffuse and declared a port (row 12) | — closed at head grain |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v135: `.../person_reflexive_dod_cross_family_direction_v135_result.json`
- v118: `.../person_dod_head_10_5_writer_fold_v118_result.json`; code `ops/run_person_dod_head_10_5_writer_fold_v118.py` (helper `ops/dod_folds.py`)
- v113: `.../person_dod_token_only_generator_v113_result.json`; code `ops/run_person_dod_token_only_generator_v113.py`
- v112: `.../person_dod_source_fold_v112_result.json`; code `ops/run_person_dod_source_fold_v112.py`
- v110 / v111: `.../person_reflexive_dod_natural_v110_result.json`, `.../person_reflexive_dod_pile_v111_result.json`; rows `person_reflexive_dod_{natural,pile}_rows_v11{0,1}.json`; miner config `ops/person_dod_natural_rows.py`
- v106: `.../person_reflexive_dod_random_set_null_v106_result.json`; v107: `.../person_reflexive_dod_response_census_v107_result.json`
- atlas v68: `bilinear_quotient/circuits/followups/atlas_reflexive_person_v68_result.json`
- v104: `.../person_reflexive_dod_battery_v104_result.json`; code `ops/run_person_reflexive_dod_battery_v104.py` (via `ops/dod_battery.py`)
