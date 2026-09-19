# Handoff — Claude circuit lane, 2026-09-18 (written 23:33 UTC)

**Read first.** `polynomial_causal/explanations/for_logan/LATEST.md` → `in_depth_circuit.md` (reflexive person, token → logit),
`in_depth_circuit_number.md` (pronoun number, the deepest component: MLP 3 → logit at two sites, compensation resolved),
`research_update_2026-09-18_unit_grain_synthesis.md` (four laws, the seven-family MLP-8 table, shared detector 829 / 1738, limits).
Cross-line log: `SHARED_READOUT_COMPONENTS.md` (table + one line per receipt). Per-line scorecards: `*_DOD_SCORECARD.md` (every prediction,
held or failed). Receipts: `bilinear_quotient/circuits/followups/*_v{164..286}_result.json`. Reviews: `REVIEW_2026-09-18_*.md` (31 today).

**Standing rules that bit today.** Every review ends with `QUEUED:` or `STOP:`; folds nominate, edits decide; quote the in-place leave-out
next to a carrier share (it predicts an edit only across adjacent blocks); never pool Δh × a mean gradient for a bilinear detector — split by
row class; write scorecard rows via `dod_scorecard_row.py --json` (no shell quoting); count natural-row files before pricing; run the
`dod_run_wait.sh` chain, never a bare wait loop.

**Tools (ops/).** `dod_units.py` — `unit_census`, `product_unit_census` (exact leave-one-out), `carrier_split`, `forward_margins` (int or list
of positions). `dod_check_runner.py` (undefined names + dry-run), `dod_run_wait.sh` (check → enqueue → wait → show), `dod_show.py`
(receipt printer), `dod_scorecard_row.py` (record step), `dod_derive.py` (AST-aware runner derivation). Miner: `dod_natural_miner.py`
records `second_offset` when a `second` set is given (verb-annotated rows v272 / v273).

**State of the collection at unit grain.** Number: 829 plural / 953 singular / 1030 weak at MLP 8, computed by MLPs 3–7 (named units at
each block), two sites (noun 5.3%, verb 5.9%, both 11.8%), OOD at MLP-8 grain at both sites (3.3% / 2.7%), 829 shared with verb agreement
(edit-decided in both), 1738 a second plural unit with opposite roles. Gender: 3152 / 3943 token-carried, two sites, 8.1's token copy at the
verb (3.5%). Determiner: 3892 a *these*-detector, half token / half MLP. Correlative: 1512 a negative-polarity detector at the final, fed by
8.1's cue copy, OOD-live, counter-case holds, not selective (v286 tests its 16.8 direction). Temporal: MLP 8 a port, MLP 7 concentrated
(1250 / 3364 / 1884; trio 2.2% in the panel; cue copy via head 7.8), not OOD (the bank is a panel construct). Person, selection: ports at
MLP 7 and 8. Copy heads: 4.5 carries the computed number feature to the verb (one axis; 7.7% whole, 3.4–4.3% axis, selective; OOD weak);
8.1 carries tokens (gender to the verb, cues to the final) across families.

**Declared limits (do not re-open without a new instrument).** MLP 1 at unit grain; the 4.5 axis and the verb-site MLP-5/6 units out of
the panel; the counter-case at the verb; the temporal trio out of the panel; 1512's selectivity (pending v286).

**Redirect candidates for a person.** A verb annotation for other families' natural rows; a natural-text bank analogue for the temporal
line; the correlative either/neither line at unit grain; Codex's regional / city path (not this lane's).

## Added 2026-09-19 03:43 UTC — MLP 1 resolved (Logan's request of ~02:00)
Full account: `for_logan/in_depth_circuit_number.md` §4.10 (facts 1–5, mechanism, units, prices, readers; receipt table v287–v327) and
`MLP1_TOKEN_TABLE_SCORECARD.md` rows T1–T41. One paragraph: MLP 1 is a per-token lookup table whose write in context equals
γ²·(entry) + (token × context cross term, which points against the entry by 0.44–0.9) + (context² term); the resulting gain α on the entry
equals the token's own-key share of attention 0/1 (r 0.65 on text). The cancellation is layer-wide (every lookup unit cancels ∝ its lookup)
with a causal head {3289, 624} (22%; replace-edit +0.152 vs census +0.169; 380× null). The raw lookup is harmful: table-everywhere costs
0.70 nats on text, more than removing MLP 1 (0.41); attention 0/1's context read exists mainly to let MLP 1 convert its lookup (all-self-only
attention → α = 1.000, +0.81 nats). Readers: block 2 (MLP 2, densely) on text; on the pronoun rows the conditioned write reaches the number
margin through the MLPs of blocks 2–3 with their attentions cancelling most of it (v325); ten MLP-3 units are the leak channel (v326/v327),
distinct from the named number units 3465/493, which read the entry's direction (v296) and are flat under un-conditioning (v322/v323).
Falsified on the way and kept: register direction (v294), context² identity (v295), re-injection floor (v299), input-identity loss (v300),
zeroing as the counterfactual (v305→v306), linear whole-layer price (v312), block patch-back as a localiser (v315), numbers-lowest (v321).

## Added 2026-09-19 04:08 UTC — the chain's units from single tokens (v329–v339)
Folded alone through blocks 0–8, the named units 3465 / 493 (MLP 3), 1036 (MLP 5), 829 (MLP 8) separate plural from singular nouns from the
token's own lookup (lexicon: 2–5 std, 100% of plurals; 400 vocabulary s-pairs: 2–3 std at 256, 82–98%); 953 / 1030 do not (lexicon-specific).
3465 reads lexical plural-NOUN sense: irregular plurals yes (0.85), -s verbs only when they have a noun reading (1.13 vs 0.09), mass / collective
nouns no, numerals no, plural pronouns no. One preceding word sharpens or suppresses them (these −4.2, two −3.5, The −3.8 vs a −1.6, and −0.5,
" 1" flips sign; 493 the complement; 829 flips after a) — agreement readers of noun number × context expectation. The 2 × 2 factor split
(v338) gave 3465 L = noun (0.81), R = licensing (0.51) but the labels did not transfer to full sentences (v339: both factors carry number; 70%
of 3465's contrast on the 'determiner' factor) — retired as wiring, kept as bilinear products. On the native rows all six separate at the noun
(2.8–7.3 std) and none at the answer position (attention carries the number, earlier chapters). Scorecard rows T43–T53; §4.10 tail.

## Added 2026-09-19 04:39 UTC — agreement at MLP 3 (v346–v353)
At a verb after a noun the chain's MLP-3 units are agreement detectors: 3465 (noun reader gated by the verb) fires −165 for "The X are" with X
singular and 493 for plural X + is; but the causal population is led by 3040 / 114 / 565 (verb-form readers gated by the noun; 3040 −1390 for
singular + are, 114 +1098 for plural + is). Zeroing 3465 alone at the verb is near-inert (KL 0.0008); zeroing the top-10 population costs 0.094
nats and 0.73 log-odds of plural continuation (nulls 0.00002). Factors are mixed in both kinds of unit (v339, v353). Rows T60–T67.
