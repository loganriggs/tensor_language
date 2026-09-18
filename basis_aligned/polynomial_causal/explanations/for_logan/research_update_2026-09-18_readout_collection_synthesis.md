# 18 September, end of night — What the readout collection says (synthesis)

Thirteen decisions in six families, each read out at the final query by a small set of late heads writing along a weight-only
direction `O_h^T(u_a − u_b)`, each taken through the same battery: fresh rows with a frozen number, a norm-matched random-direction
null, three unrelated readers, a matched-count random head-set null, keep-only sufficiency, additivity, an exact response census,
and natural rows from the training corpus and from the Pile. Numbers below are read from the receipts (`READOUT_COLLECTION.json`);
scorecards keep every failed prediction.

| family | line | heads | fresh fraction | direct share | natural FineWeb / Pile |
|---|---|---|---|---|---|
| temporal | aspectual has/had | 8.1, 9.1, 9.4 | 0.54 | — | — / — |
| temporal | temporal will/had | 11.3, 9.1, 15.5, 9.4 | 0.81 | — | — / — |
| temporal | narrative was/is |  | 0.73 | — | — / — |
| temporal | modal would/will | 9.4, 11.3, 9.1, 15.5 | 0.58 | — | — / — |
| number | lexical were/was | 11.3, 5.7, 7.8, 9.7 | 0.70 | — | — / — |
| number | perfect have/has | 11.3, 7.8, 5.3, 9.7 | 0.66 | 0.60 | 11% / 25% |
| pronoun | pronoun gender he/she | 10.1, 9.6, 12.4, 15.1 | 1.02 | 0.94 | 77% / 83% |
| pronoun | pronoun number they/he | 9.6, 12.4, 15.1, 10.5 | 0.76 | 1.03 | 65% / 56% |
| selection | adjective preposition in/of | 8.8, 6.3, 13.8, 7.8 | 0.57 | 0.38 | 12% / 12% |
| selection | verb particle up/down | 13.8, 14.8, 7.8, 8.8 | 0.43 | 0.86 | 18% / 19% |
| correlative | correlative either/not | 16.8, 14.8, 7.8, 8.1 | 0.44 | 1.17 | — / — |
| correlative | correlative both/neither | 8.1, 7.8, 16.8, 14.8 | 0.46 | 1.06 | — / — |
| person | reflexive person I/you | 8.1, 13.1, 10.5, 15.1 | 0.42 | 0.85 | 37% / 33% |
| person | object control me/you | 13.1, 8.1, 10.5, 15.1 | 0.46 | 0.90 | — / — |

**Three kinds of readout set.** (1) *Token readers*: the person set (8.1, 13.1, 15.1) and both correlative sets (8.1, 16.8, 14.8,
7.8) take 80–100% of their contrast from the cue token itself and close to a two-entry lookup with constant attention weights
(constant patterns keep 88–104% of their service); the gender set's 10.1 / 15.1 are mostly that (78%). (2) *Contextual readers of an
MLP-resolved feature*: the number sets and the pronoun-number set read a state that MLPs 8–10 wrote at the noun and verb; that
state does not fold to a few writers or pair terms, so MLP 8 (pronoun) and MLPs 8–10 / 9–17 (have/has) are declared ports.
(3) *Relayed readouts*: the temporal family, in/of and have/has act partly through the MLP suffix (40–60% of the effect), the
pronoun, person, correlative and up/down sets are direct (85–117%), with MLP 17 always pushing back.

**Three regularities that were registered the other way and falsified, kept.** On natural text, number readout sets hurt rows
whose far cue noun has the other number as much as congruent rows (four times on two lines): they carry the number the model
resolved for the real subject. Token-reader sets do the opposite — counter-case rows move toward the text's word (gender, person,
correlatives). Neither the two selection contrasts (in/of, up/down) nor the two number lines share one axis at their shared heads,
yet the temporal family does (v70): "same heads" and "same direction" are different claims, and shared heads carry different
families along orthogonal directions (cross-family removal 0.00 at three sets).

**Where the collection is honest about limits.** The number family is selective on synthetic panels but moves the tense reader on
natural text; the leak survives dropping 11.3 and orthogonalizing the readout direction and is produced downstream by MLPs 5–17
(v138–v142) — a structural number → tense dependency, declared. Head 8.1 is one token copier serving four families, with cross-talk
between the temporal and correlative lookups (v133). Twenty-eight of the atlas's 100 live lines stay outside the six cores; the
largest residues are the determiner-number variants of one task and a heterogeneous 11.3 + 7.8 + 17.4 agreement / mood cluster
that geometry assigns to the number family.

**Reports.** Families: `research_update_2026-09-18_readout_families.md` (temporal, number), `…_pronoun_readout_family.md`,
`…_selection_readout_family.md`, `…_person_readout_family.md`, `…_correlative_readout_family.md`; index
`…_readout_collection_index.md`; scorecards `basis_aligned/claude_hourly_review/*_DOD_SCORECARD.md`; cross-line summary
`SHARED_READOUT_COMPONENTS.md`; atlas `READOUT_ATLAS_TABLE.md`; receipts `basis_aligned/bilinear_quotient/circuits/followups/`.
