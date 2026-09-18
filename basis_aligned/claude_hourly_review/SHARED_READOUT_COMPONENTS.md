# Shared readout components across three auxiliary decisions (bilin18)

Cross-line summary of the definition-of-done batteries run on 2026-09-17 (Claude lane). Every number is an
edit on rows fresh to the line unless marked; receipts are listed in the three scorecards in this directory.

| line | decision | readout set (weight-only `O_h^T(u_a − u_b)`) | fresh-row margin removed | cue reader | how the cue reaches the readout |
|---|---|---|---|---|---|
| aspectual | has / had after since/by | {8.1, 9.1, 9.4} | 50–66% | **8.1** reads since/by (token-only value; 2 constants) | direct (8.1 at the final query) + relay 8.1@bank → 9.1/9.4 |
| temporal | will / had after tomorrow/earlier | {11.3, 9.1, 15.5, 9.4} | 74–83% | **8.1** reads tomorrow/earlier at the subject NP (token-only; 8 constants) | 8.1@NP + MLP8–10 → NP state → 11.3; 9.1/9.4/15.5 read cue + NP |
| narrative | was / is after "Last winter … stood" / "Every winter … stands" | {15.5, 11.3, 9.4, 9.1} | 72–74% | none at head grain (8.1 absent from the top six) | tense already in the second sentence's state; 11.3 and 9.1 read the tail |

What is shared: the block-9 pair {9.1, 9.4} relays in all three lines; 11.3 and 15.5 are the readout for both
tense/temporal lines; head 8.1 is one temporal-cue token reader whose block-0 value branch is a lookup on the
cue token, serving two lines with two tables. What is not shared: the readout contrast (a weight object per
line) and, on the narrative line, the cue path (no single cue token).

Common failure pattern: the strict four-way additivity bar (25% of the smallest single) fails on both four-head
sets by 3–4% of the joint while all pairwise terms are below gate; the largest pair term is a serial relay term
(9.1 also writes the state 11.3 reads). Common open ports: the MLP-generated parts of the relay states (diffuse
writer-pair folds, kill criterion tripped on both lines where tested).

Update 00:03 UTC (v48): on the temporal line, removing 8.1's NP write removes about 32% of the MLP8–10
part of the state 11.3 reads (the MLPs reinforce 8.1's write on that direction); the rest of the MLP part is not
attributable to any few writers at this grain. Per the review-4 kill criterion (≥ 40% required) the MLP relay ports
stay declared on both lines; no further folding of them at head/MLP grain.

## Readout families across nine decisions (reuse census, 00:15–00:20 UTC; opened authored rows, weight-only sweeps)

| decision | contrast | top-4 heads (logits) | set fraction | was−were gate |
|---|---|---|---|---|
| aspectual has/had | has−had | 9.1, 8.1, 9.4, then 15.5, 11.3 | 50–66% (fresh) | passes |
| temporal will/had | will−had | 11.3, 9.1, 15.5, 9.4 | 74–83% (fresh) | passes |
| narrative was/is | was−is | 15.5, 11.3, 9.4, 9.1 | 72–75% (fresh) | passes |
| modal would/will | would−will | 9.4, 11.3, 9.1, 15.5 | 59% | passes |
| perfect have/has | have−has | 11.3 1.54, 7.8, 5.3, 9.7 | 67% | fails (related reader) |
| lexical number were/was | were−was | 11.3 1.67, 5.7 0.60, 7.8 0.32, 9.7 0.24 | 75% | fails (related reader) |
| quantifier number was/were | was−were | 11.3 0.72, 7.8 0.57, 5.7 0.12, 13.1 0.09 | 50% | fails (related reader) |
| coordination were/was | were−was | 5.7 0.47, 11.3 0.42, 7.8 0.33, 9.7 0.09 | 49% | fails (related reader) |
| preposition on/of (non-auxiliary) | on−of | 6.3 0.49, 13.8 0.36, 7.8 0.34, 8.8 0.27 | 28% | passes |

Two readout families at the auxiliary slot: a **temporal family** {9.1, 9.4, 15.5} + 11.3 (tense, mood, aspect;
8.1 as the token-only cue reader where a single cue token exists) and a **number family** {5.7, 7.8, 9.7} + 11.3
(agreement decisions). Head 11.3 belongs to both: the general auxiliary-slot readout. The non-auxiliary contrast
(on/of) uses neither family ({6.3, 13.8, 7.8, 8.8}; 7.8 recurs as a generic helper). For number lines the
was−were reader is related, so the gate that fails there is not a selectivity failure; a number-appropriate
unrelated reader (e.g. has−had) is needed for a proper battery on that family.

## Readout-direction geometry at the shared heads (v69, weights only, 00:53 UTC)

Per head, cosines between `O_h^T(u_a − u_b)` for the contrasts has−had, will−had, is−was, will−would (temporal) and
were−was (number). At the temporal family heads the four temporal contrasts share one axis (mean |cos| 9.1 0.48, 9.4 0.65,
15.5 0.64; has−had vs is−was 0.72 / 0.84 / 0.84) and the number contrast is orthogonal to it (max |cos| ≤ 0.21).
Head 11.3 carries both: temporal mean 0.56 and number-vs-temporal 0.54 (with has−had, −0.54). The number heads 7.8 and
9.7 also mix the two (0.57, 0.41); 5.7 does not (0.16). So the two families are two weight-level axes in the late heads'
output projections: a "temporal" axis shared by four decisions and a "number" axis, entangled only at 11.3 / 7.8 / 9.7.
Receipt: `bilinear_quotient/circuits/followups/readout_geometry_v69_result.json`.

## One weight-only temporal axis for four decisions (v70, edit, fresh rows, 01:47 UTC)

At heads {9.1, 9.4, 15.5, 11.3} the first left singular vector of the four mapped contrasts (has−had, will−had, is−was,
will−would; weights only, no data) was removed on each line's fresh rows:
- aspectual: own 0.98, shared-axis 0.63 (31%, positive 50%, null max 0.02, selective True)
- temporal: own 3.45, shared-axis 2.47 (58%, positive 100%, null max 0.05, selective True)
- narrative: own 1.46, shared-axis 1.16 (58%, positive 100%, null max 0.02, selective True)
- modal: own 1.60, shared-axis 1.45 (52%, positive 100%, null max 0.04, selective True)

The shared axis carries ≥ 50% of each line's own-contrast removal on all four lines (registered pred_c held); it is live
and selective on temporal, narrative and modal; on the aspectual line the registered LIVE/null bar failed (that line's own
set includes 8.1, excluded here). Receipt: `bilinear_quotient/circuits/followups/temporal_axis_v70_result.json`.

## Third family: pronoun features (atlas v68 complete, v71 fresh rows, 01:57 UTC)

The atlas finished at 225/226 scored lines (118 capable, 100 capable-and-live; `readout_atlas_v68_result.json`). Among live
lines the most recurrent top-4 core is {9.6, 12.4, 15.1} (19 lines), all pronoun gender / number / person decisions; next
{11.3, 6.3, 7.8} (12) and {11.3, 17.4, 7.8} (11), both number-family variants. Pronoun gender he/she on fresh rows (v71):
set {10.1, 9.6, 12.4, 15.1} removes 0.96–1.11 of the margin, positive 60/60, null max 0.02, selective, additive (gap at
the bar), keep-only retention 1.37; the frozen band 0.89 ± 0.15 failed upward on one frame. Scorecard
`PRONOUN_GENDER_DOD_SCORECARD.md`.

| family | core heads | lines (atlas, live) | fresh-row battery |
|---|---|---|---|
| temporal | 9.1, 9.4, 15.5 (+11.3, +8.1 cue reader) | aspectual, temporal, narrative, modal | all four |
| number | 5.7, 7.8, 9.7 (+11.3) | perfect, lexical, quantifier, coordination + atlas variants | number-within-scope |
| pronoun | 9.6, 12.4, 15.1 (+10.1 on gender) | 19 pronoun gender/number/person lines | pronoun gender (v71 fresh, v72 random-set null, v73 natural FineWeb 77%, v74 response census 94% direct, v75 Pile 83%); pronoun number they/he (v76 fresh rows 7/7: set {9.6, 12.4, 15.1, 10.5} 76%, frozen 0.71 ± 0.15 held, selective, additive, keep-only 1.53) ; v79 random-set null 4/4; v77/v78 natural FineWeb 65% / Pile 56% on congruent rows, counter-case prediction falsified twice: the number set reads a resolved number, not the noun — the core carries a second pronoun decision |

v80 source fold (both pronoun lines, fresh rows): head 10.1 is a token-only gendered-noun reader (98% noun, 72% block-0 value) — the pronoun family's analogue of the temporal family's 8.1; 15.1 half token-only; 9.6 / 12.4 contextual. The number set is contextual throughout (token-only ≤ 11%). Registered predictions c and e failed (kept).

v82 writer fold (9.6's sources, both pronoun lines): the noun and verb states are MLP-written, MLP 8 first (0.27–0.54), MLP 6 second; head 8.1 writes 20% of the gender feature at the verb. MLP 8 is the shared open port across the pronoun lines and the auxiliary families' relay parts.
v83 pair fold of MLP 8 (both pronoun lines): diffuse (largest pair 15% / 7%); MLP 8 declared port. Both pronoun lines closed at head grain (02:23 UTC).

## Atlas family assignment (02:26 UTC, `atlas_summary.py` FAMILIES, ≥ 2 core heads in the top-4)

100 capable live lines: pronoun 28, number 17, temporal 3, other 52. Among the 52 "other": 16 lines carry 11.3 + 7.8 (with 17.4 or 6.3) — auxiliary
agreement lines the number core {5.7, 7.8, 9.7} under-counts because they use 11.3 instead of 5.7/9.7 (adjective_finiteness, adjective_subjunctive, agreement_do_does, agreement_lifts_lift, comparative_complement_inferior, dative_alternation, finiteness_selection, for_complementizer_waited …);
9 carry 13.8 + 7.8 (adjective_preposition_in_of, benefactive_preposition_give, comparative_complement_from_than, comparative_complement_inferior, dative_alternation, for_complementizer_waited, identical_distinct_preposition, verb_particle_up_down, verb_preposition); 5 carry 10.5 + 15.1 + 8.1 (possessive_disjoint_my_your, reflexive_object_control_plural, reflexive_object_control, reflexive_person_plural, reflexive_person). The temporal family's low count (3) is
because its own four lines are not atlas lines (the atlas ran the corpus candidates, not the lane's authored panels) and most
tense/mood atlas cells were incapable or not live. The table with the family column: `READOUT_ATLAS_TABLE.md`.

## Fourth family candidate: complement selection (v85, 02:29 UTC)

Adjective preposition in/of on fresh rows: set {8.8, 6.3, 13.8, 7.8} 57% (frozen 0.54 ± 0.15 held in all three frames), positive 96/96,
null max 0.06, selective by the gate (night−day 4× null, stated), additive, keep-only 1.08 — 7/7. The same four heads were v54's
on/of set. Scorecard `SELECTION_DOD_SCORECARD.md`. Family test next: verb particle up/down on the shared core {13.8, 7.8, 8.8}.

| family | core heads | lines (atlas, live) | fresh-row battery |
|---|---|---|---|
| selection | 13.8, 7.8, 8.8 (+6.3, 14.8) | 9 complement / particle / preposition lines + v54 | adjective preposition in/of (v85 7/7), verb particle up/down (v86 7/7) — a component family |
v87–v90: both selection sets beat random quadruples; up/down is direct (86%), in/of is MLP-relayed (38% direct; MLPs 8–15 amplify) — the same relay shape as the temporal family, in a family that starts at block 6.
v91–v94 natural rows (FineWeb, Pile) for both selection lines: bars held, fractions 12% (in/of) and 18–19% (up/down) of bigram-sized natural margins; in/of far-cue counter-cells falsified twice (kept). Selection family now has fresh, natural in- and out-of-corpus, random-set null and response census on both lines (02:36 UTC).

## Cross-family geometry at the shared heads (v95, weights only, 02:38 UTC)

`family_geometry_v95_result.json`, 3/5. Held: the two pronoun contrasts share an axis at the pronoun core (|cos(he−she, they−he)| 0.43 / 0.55 / 0.38
at 9.6 / 12.4 / 15.1; −0.41 already at the unembedding); number and selection are orthogonal at 7.8 (|cos| ≤ 0.25); pronoun and temporal
are orthogonal at 15.1. Failed and kept: the two selection contrasts (in−of, up−down) do NOT share an axis at 13.8 / 8.8 (|cos| < 0.2) — the
selection family shares heads, not a direction; and the cross-family unembedding bound (0.25) was exceeded by were−was | has−had (−0.26, the
number/aspect entanglement known from v56/v57). Observed, not registered: they−he aligns with were−was at 9.6 (0.38), 15.1 (0.40), 8.8 (0.47),
11.3 (0.40) — a candidate shared NUMBER axis between pronoun number and verb agreement; tested by removal (v96): the were−was direction at the pronoun-number heads removes 18% of what the own direction removes (live, null-beating, gender direction below it) — the registered half failed; a small shared component, not one axis (unlike the temporal family's v70).

v97 (02:43 UTC): the number family's second line at fresh grain — perfect have/has, set {11.3, 7.8, 5.3, 9.7}: 66% (frozen 0.66 ± 0.15, landed at 0.66), positive 96/96, null 0.04, selective on non-number readers, keep-only 1.03; additivity FAILED (joint 2.11 vs singles 1.86, bar 0.02 because 5.3 is 0.09). Scorecard `PERFECT_NUMBER_DOD_SCORECARD.md`.
v98–v101 (perfect have/has): random-set null 4/4; census 60% direct with an MLP 9–17 relay; natural FineWeb 11% / Pile 25% of the margin, positive 32/32 and 30/32; the counter-case prediction (far noun of the other number) failed for the third and fourth time on a number line: number readout sets carry the RESOLVED subject number (v77, v78, v100, v101), the gender set tracks the noun (v73, v75).
v102 (have/has pairwise): the additivity failure is a serial relay — pair terms with 11.3 carry 82% of the interaction, largest 11.3 × 7.8; higher-order terms negligible. Same shape as the temporal family's 9.1 → 11.3 relay (review 6).
v103 (have/has folds): 11.3 reads the head noun (0.60) and the preposition (0.39); the noun state on its reader direction is 85% MLP-written (MLPs 8/9/10), heads 4% — the v102 pair synergy is not a head-to-head relay but a downstream MLP one. Perfect have/has closed at head grain with MLPs 8–10 / 9–17 as ports (02:51 UTC). Every family now has two lines closed at head grain.

## Fifth family: person (v104 / v105, 03:05 UTC)

Reflexive person I/you (subject antecedent) and object-control me/you (object antecedent) both read out at {8.1, 13.1, 10.5, 15.1} along
`O_h^T(u_myself − u_yourself)`: 42% and 46% on fresh rows (frozen 0.46 / 0.54 ± 0.15 held), positive 96/96 each, null ≤ 0.02, selective,
additive, keep-only 1.21 / 1.29 — 14/14. Scorecards `PERSON_REFLEXIVE_DOD_SCORECARD.md`, `PERSON_OBJECT_CONTROL_DOD_SCORECARD.md`.

| family | core heads | lines (atlas, live) | fresh-row battery |
|---|---|---|---|
| person | 8.1, 13.1, 10.5, 15.1 | 5 reflexive / possessive person lines | reflexive I/you (v104 7/7), object control me/you (v105 7/7) |
v106–v109: both person sets beat random quadruples (2.27 vs 0.10; 2.23 vs 0.15) and are direct readouts (85% / 90% of the effect in the heads' own writes; MLP 17 counteracts). Person natural panels mined (FineWeb 20000 docs: I/myself 16, you/yourself 16, counter-cases 10 + 1; Pile: 16 / 16 / 3 + 1).
v110/v111 (reflexive person natural rows): 37% / 33% of the natural margin, positive 32/32 on both corpora, selective; counter-cases shift toward the text — the person set tracks the cue token (token-reader family: 8.1 / 10.1 shape), unlike the number sets. Person family: fresh, natural ×2, null, census on both lines by 03:07 UTC.
v112 (reflexive person source fold): the person set is a TOKEN-ONLY reader family — 8.1 / 13.1 / 15.1 take 80–98% of their contrast from the I / you position, 72–95% through the block-0 value branch (10.5 is the contextual member). Three families by source type: token readers (person; gender's 10.1 / 15.1; temporal's 8.1), contextual readers of MLP-resolved state (number, pronoun number, selection in/of via MLPs), and mixed (gender's 9.6 / 12.4).
v113: the three person token readers are a token-only generator — constant per-(head, cue) patterns from other constructions retain 94% of their service (0.83–1.12 per construction), native pattern 91%. Ports closed for 8.1 / 13.1 / 15.1 on this line; 10.5 open. The temporal line's 8.1 (v12) and the person line's 8.1 / 13.1 / 15.1 are the same mechanism on different cue tokens.

## Atlas coverage with five cores (03:13 UTC)

Of 100 capable live lines: pronoun 26, number 17, person 13, selection 13, temporal 3 — 72 assigned; 28 residue. The residue is structured:
{11.3, 7.8, 17.4} on 7 verb-inflection lines (partitive agreement 0.55, do/does 0.44, adjective finiteness 0.44, subjunctives 0.32–0.39,
finiteness selection 0.33, lifts/lift 0.29, modal perfect 0.15) — a sixth-family candidate on 11.3 + 7.8 with 17.4 as the new head;
{11.2, 5.7, 16.8, 15.1} on the four determiner-number crates variants (one task); {16.8, 14.8, 8.1, 7.8} on the correlative lines (0.43–0.55).
Table with the family column regenerated: `READOUT_ATLAS_TABLE.md`.
v115 (object-control source fold): identical token-reader structure on the family's second line (8.1 / 13.1 95% token-only from me / you, 15.1 64%, 10.5 contextual) — the person family's mechanism replicates across subject and object antecedents.

## The residue cluster {11.3, 7.8, 17.4} is the number family, not a sixth family (v116, weights only, 03:14 UTC)

`inflection_cluster_geometry_v116_result.json`, 1/4. The two number contrasts of the cluster (has−have, do−does) share one axis at 11.3 / 7.8 / 17.4
(|cos| 0.95 / 0.86 / 0.78) and align with were−was (0.78 / 0.58 at 11.3 / 7.8): the agreement lines are number-family lines with 17.4 as a
member the batteried sets did not include. Failed and kept: the mood contrast be−was is not orthogonal to number at those heads (|cos| ≈ 0.5),
finiteness (to−that) and mood do not share an axis (to−that is orthogonal to everything, < 0.2), and the cross-type unembedding cosines reach
0.39 (has−have | be−was −0.37; has−have | do−does −0.65 within type) — part of the head-level alignment is inherited from the vocabulary. Reading:
the mood lines ride the number / tense axes at the shared heads; the finiteness lines are a separate, weak decision. The number core in the atlas
assignment is now {11.3, 5.7, 7.8, 9.7} (the union of the two batteried sets), which absorbs the agreement lines.
Assignment after widening the number core (03:15): number 35, pronoun 26, other 15, person 13, selection 8, temporal 3 (85 of 100 assigned). Caveat: five selection lines whose top-4 holds 11.3 + 7.8 as well as 13.8 + 8.8 now tie at two core heads each and go to number by the tie rule (family order); the selection family's own count is 13 by the earlier rule. The residue of 15 is the determiner-number crates variants {11.2, 5.7, 16.8, 15.1} and the correlative lines {16.8, 14.8, 8.1}.
v117: the object-control line's token readers are also a token-only generator (native 94%, constant patterns 94% pooled, CV ≤ 0.15). Person family closed at head grain on both lines with three of four heads closed to tokens + constant patterns (03:16 UTC).
v118: 10.5's source at the pronoun (person line) is diffuse — embedding 25%, MLPs 45% (largest 13%), heads 30% — declared port; the reflexive-person line is closed at head grain with three heads closed to tokens. Correlative cluster {8.1, 16.8, 14.8, 7.8} (5 atlas lines, 8.1 leading on a token cue) under test as a sixth family (v119 either/not → or/but, v120 both/neither → and/nor).

## Sixth family: correlatives (v119 / v120, 03:24 UTC)

either/not → or/but (v119, 7/7: 44% vs frozen 0.43, positive 96/96, null 0.06, selective, additive, keep-only 1.14) and both/neither → and/nor (v120, 6/7:
46% vs frozen 0.47, positive 96/96, null 0.09, selective, keep-only 0.96; additivity failed, over-additive with 8.1 at 1.22) at {8.1, 16.8, 14.8, 7.8}.
Readers as run were will−would / who−which / night−day (an import side effect: the runners import v104, which sets L.READERS at module level;
corrected in the docstrings and recorded here). Scorecards `CORRELATIVE_EITHER_NOT_DOD_SCORECARD.md`, `CORRELATIVE_BOTH_NEITHER_DOD_SCORECARD.md`.

| family | core heads | lines (atlas, live) | fresh-row battery |
|---|---|---|---|
| correlative | 8.1, 16.8, 14.8, 7.8 | 5 correlative lines | either/not (v119 7/7), both/neither (v120 6/7) |
v121–v124 (correlatives): both sets beat random quadruples (1.99 vs 0.20; 3.37 vs 0.10) and are direct readouts (117% / 106% — the late MLPs 15–17 push back by +1.0 / +1.4 logits, the strongest counter-response in the collection). First runners emitted by `dod_line.py`. Natural correlative panels mined (either/not full 4 × 16 on both corpora; both/neither lacks both/nor rows: 0 FineWeb, 2 Pile).
v125–v128 (correlative natural rows, 24/24 predictions): either/or 29–34% and neither/nor 116–130% of their natural margins (the set carries the whole 'nor' decision), not/but 5–14% and both/and 5–6% (weak or default-dominated halves); counter-cases move toward the text on all four panels — a token-tracking family like person and gender. Six families with natural in-/out-of-corpus evidence on ≥ 1 line each (03:28 UTC).
v129/v130 (correlative source folds, 10/10): all four heads {8.1, 16.8, 14.8, 7.8} read the correlative's first element from its token (98–101% of each contrast at the cue position; token-only branch 76–99% for 8.1 / 14.8 / 16.8, 56–59% for 7.8). Families by source type now: token readers (correlative, person, gender's 10.1 / 15.1, temporal's 8.1), contextual (number, pronoun number, selection in/of), mixed (gender 9.6 / 12.4, selection up/down).
v131/v132: both correlative sets close to token-only generators with constant patterns (native 84% / 96%, constant 88% / 104% pooled, ≥ 0.70 per held-out construction); the pattern-stability bar (CV ≤ 0.35) failed on both (0.48 / 0.36) and is kept. Correlative family closed at head grain, 03:31 UTC. Six families, thirteen lines, all closed at head grain.

## Head 8.1 as one token lookup across families (v133, weights + single-token block-0 values, 03:33 UTC)

`head_8_1_lookup_table_v133_result.json`, 2/3. For every registered cue pair the sign of (u_pos − u_neg) · O_{8.1} (v1(a) − v1(b)) matches the
decision (since/by → has−had 559; tomorrow/earlier → will−had 1417; I/you and me/you → myself−yourself 3807 / 3795; either/not → or−but 1651;
both/neither → and−nor 2570), and each beats the largest of 16 random token pairs on its contrast (random max 314–1004). Exploratory: king/queen →
he−she 1648 vs random max 546 — 8.1 also serves the gender readout it was fifth for in the sweep. Failed and kept: contrast specificity — the
either/not lookup pushes and−nor by 0.35 of its own score, since/by pushes the person and correlative contrasts by 0.37, tomorrow/earlier pushes
or−but by 0.27; the person and both/neither lookups are specific (≤ 0.10). Reading: 8.1 copies each cue token's block-0 value to the final query
once; the families read that one write along their own directions, which are not orthogonal for the temporal and correlative cues.
v134 (shared heads, separate directions, 5/5): at the pronoun-number set the person family's myself−yourself direction removes 0.00 (own 1.55; null 0.01). Heads shared between families (15.1, 10.5 here; 7.8 between number and selection, v95) carry each family along its own direction — the collection's components overlap in heads, not in what they write.
