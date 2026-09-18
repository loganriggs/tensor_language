# Research update — 18 Sep 2026: the readout collection at unit grain

*Claude circuit lane, written 19:35 UTC after review 27. Every number is read from a receipt under
`basis_aligned/bilinear_quotient/circuits/followups/` (named in brackets); the per-line scorecards under `basis_aligned/claude_hourly_review/`
hold every registered prediction, held or failed. Anchors: `in_depth_circuit.md` (reflexive person, token → logit) and
`in_depth_circuit_number.md` (pronoun number, token → logit, the deepest component).*

## 1. What changed today

This morning the collection was sixteen readout components in seven families, each a set of heads with a weight-only direction, tested
at head grain (fresh rows, random-direction null, three unrelated readers, natural rows). Today the MLP "ports" those components read
were opened to **unit grain** with two exact tools — a per-unit census of an MLP's write on a reader direction, and a carrier split of a
bilinear unit's contrast into its writers' own changes — and every nomination was then decided by an edit against a random-unit null.

## 2. Four laws that held across receipts

1. **Mass is not carriage** [v185–v190]. A writer can hold a large share of a bilinear unit's pair terms while its write is identical
   across the contrast (head 6.3 held 32% of unit 2483; 89% of that was reader-borne). The carrier identity — own change × partner mean,
   summing to one — reverses the gender story: both gender detectors are token-carried (~70%), the number detector is MLP-carried (66%).
2. **Bilinear detectors respond class-wise** [v238, v239, v247]. A relay's shift can *restore* a detector it feeds because the detector's
   gradient along that relay has opposite signs on the two row classes. Never pool Δh × a mean gradient; sum gradient × shift per class.
   This is the whole of the "compensation" that made edits run 2–4× below carrier shares (MLP-7 unit 1779), after four wrong readings of
   mine were falsified and kept [v230–v237].
3. **Read from the token vs computed** [v189–v193, v201, v256]. Gender is carried by the noun embedding and its self-copies (heads 6.x,
   8.1); number is computed by MLPs 1–7 with the embedding never above 20% at any stage; determiner number is half and half. The
   distinction reaches the verb site: gender travels with its token (8.1's copy), number must be copied as a feature (head 4.5), and
   the 2×2 of edits is clean (4.5: −7.7% number / −0.5% gender; 8.1: −3.5% gender / −0.2% number) [v206, v219, v221, v223].
4. **Detectors are shared; reader weights are not** [v266–v270]. MLP-8 unit 829 (plural) leads the number write for pronoun choice *and*
   verb agreement, edit-decided in both (5.3% with its trio; 1.9% alone on have/has) and plural-only on natural text in both. Unit 1738 is a
   second plural detector that supports *have* (−1.3% when removed) and opposes *they* (+0.5% when removed).

## 3. Seven families at the MLP-8 stage

| family | MLP 8 on the reader direction | named units | carried by | edit |
|---|---|---|---|---|
| pronoun number | concentrated (top-10 78–89%) | 829 plural, 953 singular, 1030 weak | MLPs 3–7 (computed) | 5.3% noun, 5.9% verb, 11.8% both |
| pronoun gender | concentrated (~90%) | 3152 male, 3943 female | the token (8.1 / 6.x copies) | 2.6% noun, 7.8% all |
| noun-number (these/this) | one unit 93% | 3892 a *these*-detector | half token, half MLPs 4–7 | 0.7% cue, 1.3% all |
| perfect-number (have/has) | concentrated (88%) | **829 shared**, 1738, 3858 | as 829 | 1.9% / 3.0%; natural plural-only |
| correlative (both/neither) | spread at the cue; **final** one unit 59% | 1512 negative-polarity | 8.1's cue copy at the final | 1.8% / 2.5%; natural 1.9%, not selective |
| temporal (has/had) | MLP 8 spread (19%); **MLP 7 concentrated** at the bank (top-10 73%) | MLP-7 unit 1250, a *since*-detector (32/32); 3364, 1884 | 8.1's cue copy + MLP 7 | 0.6% for 1250 [v278, v280]; MLP-8 unit 13 inert |
| person (I/you) | spread, frame-specific (26%) | none | token copies | port |
| selection (particle) | spread at both positions (29% / 8%) | none | — | port |

Content-word features have detectors at MLP 8; function-word cues mostly do not there — but the temporal family's detector turned up one block down, in MLP 7 (unit 1250, a *since*-context detector at the bank the readers read [v278, v280]), so "port at MLP 8" means "look one block lower", not "no unit". The adjective cue of the selection line is the exception to the simplest form of the content/function rule. Every detector is small under edit (0.7–5% of a margin) unless two sites add.

## 4. The deepest component

The pronoun-number line is named at unit grain from MLP 3 to the logit, at two sites (noun and verb, 5.3% + 5.9% = 11.8% together), with
head 4.5 carrying the computed feature to the verb along one 128-d axis whose *sign* is the number [v205–v215], and the MLP-8 detectors
live out of the panel at three stages (3.3% → 0.8% → 0.5%, each ≥ 40× its null, counter-case holding) [v197–v200]. Full account:
`in_depth_circuit_number.md`.

## 5. Declared limits

MLP 1 is a port at unit grain (a spread computation) [v202]. The 4.5 number axis is weak out of the panel (0.8% at twice the matched
null) and the verb-site MLP-5/6 units are undetectable there without a verb annotation [v212, v248]. Unit 1512 passes every test but
selectivity — it is a shared polarity unit, not a correlative component [v264b]. Carrier shares are first-order: the in-place leave-out
predicts an edit only across adjacent blocks [v225, v226]. Today's receipts: v164–v270; 36% of registered predictions failed and are kept.
