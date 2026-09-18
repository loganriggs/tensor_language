# 18 September — Person: a fifth readout family, {8.1, 13.1, 10.5, 15.1}

**Path now.** A first- or second-person antecedent (I / you as subject; me / you as object of a control verb) is read out at
the reflexive by four late heads writing along `O_h^T(u_myself − u_yourself)` at the token before the reflexive. Five live atlas
lines share the set (reflexive person, its plural, object control, its plural, possessive my/your). Head 8.1 leads or co-leads:
the same head that is the temporal family's token-only cue reader, here reading a person-marked pronoun token. **Delta since
the selection update:** two lines through the full head-grain battery.

| line (edit) | set | fresh rows: fraction vs frozen, positive, null | random-set null | census (direct share) | natural FineWeb / Pile (congruent) |
|---|---|---|---|---|---|
| reflexive I/you (v104, v106, v107, v110, v111) | 8.1, 13.1, 10.5, 15.1 | 0.47 / 0.40 / 0.40 vs 0.46 ± 0.15; 96/96; 0.02 | 2.27 vs 0.10, none live | 85% | 37% and 33% of 5.9–6.3 logits, 32/32 positive on both |
| object control me/you (v105, v108, v109) | 13.1, 8.1, 10.5, 15.1 | 0.44 / 0.43 / 0.50 vs 0.54 ± 0.15; 96/96; 0.01 | 2.23 vs 0.15, none live | 90% | — |

All registered predictions held (35 of 35), including the counter-case reading on natural rows: the few rows where a quoted
speaker's reflexive disagrees with the nearest I / you move toward the text's word under removal, so the set follows the cue
token — the behaviour of the gender set (token readers 10.1, 15.1) and not of the number sets (resolved number). Both sets are
direct readouts (MLP 17 pushes back by +0.3 logits), selective against tense (will−would), animacy and the canonical reader,
additive, and sufficient at their own boundary (keep-only retention 1.21 / 1.29).

**Where the heads read (v112, fold, exact).** Three of the four heads are token readers of the I / you token: 8.1 and 13.1 take
98% of their contrast from that position, 91% and 95% of it through the block-0 value branch (the token's own value, no
context); 15.1 80% / 72%; 10.5 is the contextual member (52% from the pronoun, 29% token-only, the rest from the final and
other positions). The registered "token-only share ≤ 0.50" failed (0.82) and is kept. This is why the set follows the cue on
natural text: it copies the pronoun token. The open port is the heads' attention pattern (which position they copy), the
same port the temporal line's 8.1 had, and the next step is the token-only generator with a constant pattern.

**Closed to tokens (v113, edit).** Replacing the three token readers' writes by the token-only term p × λ × v1(pronoun) keeps 91% of
their service with the native pattern and 94% with a constant pattern per head and cue taken from the other constructions
(0.90 / 1.12 / 0.83 per held-out construction; pattern variation ≤ 26%). Three of the four heads are therefore a two-entry
lookup table on the pronoun token with a constant attention weight — the temporal line's 8.1 mechanism (v12) again — and
their ports are closed; 10.5 remains a contextual reader.

**The second line reads the same way (v115).** On the object-control rows 8.1 and 13.1 take 98–101% of their contrast from the me / you
token (95% token-only), 15.1 76% / 64%, 10.5 52% / 28% — five of five registered readings, so the family's mechanism is one
thing across subject and object antecedents.

**And closes the same way (v117).** On the object-control line the constant-pattern token-only generator keeps 94% of the three heads'
service (0.82 / 1.29 / 0.81 per held-out construction, pattern variation ≤ 15%).

**Scope.** Two behaviours, fresh panels, one natural pair, head grain. Not yet: 10.5's source on either line, the three remaining lines of the cluster. Scorecards: `basis_aligned/claude_hourly_review/PERSON_REFLEXIVE_DOD_SCORECARD.md`,
`PERSON_OBJECT_CONTROL_DOD_SCORECARD.md`; receipts under `basis_aligned/bilinear_quotient/circuits/followups/person_*`.
