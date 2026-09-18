# Demonstrative number these/this (ones vs one) — definition-of-done scorecard

Opened 2026-09-18 09:35 UTC (Claude lane; a seventh-family CANDIDATE — noun number, one line so far). Rubric: `better_circuits.md` §1.

**Path now.** A demonstrative's number (these / this) across an adjective → readout set {11.2, 7.8, 6.3, 15.1} along `O_h^T(u_ones − u_one)` at the
adjective → ones / one. The atlas residue's last structured cluster: determiner several/each → crates/crate (0.18, 0.20), numeral three/one (0.22) and
this line (0.28) all lead with head 11.2 (0.40–0.66 alone). Readers set explicitly and number-free: will−would, who−which, night−day.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Blind sweep on the authored rows: 11.2 0.66, 7.8 0.14, 6.3 0.12, 15.1 0.11 (next 9.7 0.08); top-4 set 28%, live, selective | edit | opened (atlas v68) | | established |
| 2 | Frozen 0.28 ± 0.15 on fresh rows (16 fresh agents, 16 fresh objects, 16 fresh adjectives; three unused frames) | edit, frozen | fresh (v157) | fractions wanted 0.30 / kept 0.35 / counted 0.33 (pooled 0.33); positive 96/96; null max 0.07; capability 1.00 in all six cells | passes |
| 3 | Selective (will−would, who−which, night−day) | edit | fresh (v157) | moves 0.09 / 0.09 / 0.10 vs null 0.06 / 0.09 / 0.08 | passes |
| 4 | Additive | edit | fresh (v157) | singles 11.2 0.74 / 7.8 0.17 / 6.3 0.13 / 15.1 0.13; gap 0.011 ≤ bar 0.033 | passes |
| 5 | Keep-only readout at the four heads keeps their service | edit | fresh (v157) | zero 1.92; retention 0.79; random keep ≤ 0.02 | passes |
| 6 | Matched-count random four-head-set null | edit | v157 rows (v159) | set 1.18 (fraction 0.33) vs random max 0.02 (fraction 0.005); none live | passes 4/4 |
| 7 | Response census (exact λ-recurrence split from block 6, closure 6e-6): the set's own writes carry 84% (attn:11 −0.75, attn:15 −0.13, attn:07 −0.09 of −1.18); downstream net −0.19 (MLP 17 −0.13, MLP 16 −0.09 amplify; MLP 13 +0.08); remainder 0.6% | response | fresh (v160) | | passes 5/5 — a direct readout, 11.2's write dominant |
| U1 | MLP 8 at UNIT grain on head 11.2's ones − one reader direction (v157 rows, pooled these − this; closure 0.0): at the CUE (these / this) one unit dominates — 3892 carries 93% of MLP 8's write, 684 opposes (−26%), 1738 +4%; 829 (the pronoun line's plural detector) is rank 4 at ~2%, 953 rank 1269; top-10 81%; Jaccard with the pronoun-number top-50 (v168) 0.06. At the FINAL: 3892 75%, 829 rank 3 (4%), 953 rank 13 | fold | v157 rows (v254) | closure, "829 or 953 in the cue top-5" (829 rank 4), "top-10 ≥ 0.50" held; "Jaccard ≥ 1/3" false (0.06) | 3/4 — the noun-number family has its OWN MLP-8 detector, unit 3892 (a plural-determiner detector), not the pronoun line's 829 / 953: number is represented by different MLP-8 units for determiners and for nouns, with 829 a small shared contributor. Composes at unit grain is partial: shared reader heads (11.2 / 15.1 with the pronoun set), separate detectors |
| U2 | MLP-8 unit 3892 examined (v157 rows): at the cue u_3892 = −135 on these-rows vs +7 on this-rows (sign agreement 48/48); the cue contrast is 2.2× the final's; cos(Down8[:,3892], r_11.2) = −0.32; zeroing 3892 at the cue changes the pooled ones − one margin by −0.66% (all positions −1.3%); 16 random single units 0.00% | fold + edit | v157 rows (v255) | "separates these/this", "Down aligns ≥ 0.30", "fires at the cue ≥ 1.5×" held; "edit ≥ 0.05 and beats random" false as a compound (beats random: yes; ≥ 0.05: no, 0.0066) | 3/4 — unit 3892 is a nameable PLURAL-DETERMINER detector on the path (its write projects on 11.2's reader), real under edit and small (0.7%): MLP 8 is a minor input to 11.2 here, as it is to 9.6 on the pronoun lines (2.6–5.3%) |
| U3 | CARRIER split of unit 3892's these − this contrast at the cue (exact, closure held, shares sum to 1.000): embedding 38% (the largest single carrier), mlp:07 22%, attn:07 11%, mlp:04 8%, mlp:06 6%, mlp:05 4%; MLP stack 46%, attention 17% | fold | v157 rows (v256) | closure, "attention ≤ 0.20", "embedding is the top carrier" held; "embedding ≥ 0.40" (0.38) and "MLPs ≤ 0.40" (0.46) false | 3/5 — the plural-determiner detector is HALF read from the token and half computed: between the gender detectors (token ~70%) and the number detector (MLP 66%). A function word's number is partly in its embedding and partly rebuilt by MLPs 4–7 |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain (row 6) | — |
| Predicts OOD | held on a fresh panel with a frozen number (row 2) | natural rows (these / this + adjective → ones / one is rare in text; a numeral or determiner line may mine better) |
| Extracted | held at the head boundary (row 5); direct readout (row 7) | source fold of 11.2 |
| Selective | held (row 3) | — |
| Composes | additive (row 4) | — |

## Receipts
- v159: `.../noun_number_demonstrative_dod_random_set_null_v159_result.json`; v160: `.../noun_number_demonstrative_dod_response_census_v160_result.json` (both emitted by `ops/dod_line.py`)
- atlas v68: `bilinear_quotient/circuits/followups/atlas_demonstrative_number_v68_result.json`
- v157: `.../noun_number_demonstrative_dod_battery_v157_result.json`; code `ops/run_noun_number_demonstrative_dod_battery_v157.py`
