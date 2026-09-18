# A second circuit, input to logit: the pronoun-number component in bilin18

*Claude circuit lane, 2026-09-18 (afternoon). The line: "The **traders** lost the coin and so" → ` they`; "The **trader** lost the coin and so"
→ ` he`. Every number is read from a receipt under `basis_aligned/bilinear_quotient/circuits/followups/` (named in brackets); every
registered prediction, held or failed, is on `basis_aligned/claude_hourly_review/PRONOUN_NUMBER_DOD_SCORECARD.md` (rows 1–37). This is the
companion to `in_depth_circuit.md`: that component was a token copied by three heads; this one is a feature **computed** by a stack of
bilinear MLP units and read by four heads, and it is the deepest path in the collection — named units at MLPs 3, 4, 5, 6 and 8, each
stage tested by an edit against a random-unit null, and the top three stages tested on natural text.*

## 0. Why this one

The gender component (`in_depth_circuit.md`, appendix) showed the MLP port opens to named units, but its detectors turned out to be
carried by the noun token itself. The number component is the case where the port opens onto a *computation*: at every stage from
block 1 to block 8 the plural − singular contrast is carried by earlier MLPs, and the raw embedding direction never carries more than
20% of it. It is also the component where the unit-grain story is **more selective than the heads that read it**: on natural text the
head set failed the counter-case (removing it did not help the number-incongruent pronoun), while the MLP-8 units pass it at every stage.

## 1. The behaviour and the rows

Fresh rows: 16 agent nouns × 16 objects × three frames, plural (` they`) vs singular (` he`), 96 rows, 48 aligned pairs [v76]:

| frame | plural → ` they` | singular → ` he` |
|---|---|---|
| lost | The traders lost the coin and so | The trader lost the coin and so |
| because | Because the traders wanted the coin, | Because the trader wanted the coin, |
| later | The traders bought the coin and later | The trader bought the coin and later |

Capability 94–100% per cell; the native oriented margin $\ell(\text{they})-\ell(\text{he})$ averages 2.05 logits. The noun is a single
GPT-2 token in both forms (` traders` = 21703, ` trader`), at position 1 in the *lost* and *later* frames and position 2 in *because*; the
readout is at the last position (6). So at the noun position the only sources are the sentence start and the noun itself: whatever the
model knows about number there, it computed from the token.

## 2. Model facts this circuit uses

Beyond §2 of `in_depth_circuit.md` (squared bilinear attention, block-0 value mixing, the residual recurrence $x_{\ell+1}=\lambda^{(0)}_\ell
x_\ell+\lambda^{(1)}_\ell x_0+\text{attn}_\ell+\text{mlp}_\ell$, logits $30\tanh(W\,\text{rms}(x_{18})/30)$), one fact carries this document:

**Bilinear MLP units are products.** With $\hat x=\text{rms}(x)$ the block input, unit $j$ of an MLP is

$$u_j \;=\; (L_j\cdot\hat x)\,(R_j\cdot\hat x),\qquad \text{mlp}(x)=\sum_j u_j\,\text{Down}_{:,j}+b .$$

Because $\hat x=\sum_w C_w/\text{rms}$ is a sum over the writers that produced it (embedding, every earlier attention and MLP block, the
current block's heads), the unit is a sum of **pair terms**,

$$u_j=\sum_{a,b} T_{ab},\qquad T_{ab}=\frac{(L_j\cdot C_a)(R_j\cdot C_b)}{\text{rms}^2},$$

and a writer can matter in two different ways. Its **mass** is the sum of pair terms it sits in. Its **carriage** is the part of the
plural − singular *contrast* that exists because *its own write changed* between the two members of a pair: with $a_w=(L_j\cdot
C_w)/\text{rms}$, $b_w=(R_j\cdot C_w)/\text{rms}$, $A=\sum_w a_w$, $B=\sum_w b_w$, and $\Delta$, $\bar{\ }$ the pair difference and mean,

$$u_j^{P}-u_j^{S} \;=\; \sum_w\big(\Delta a_w\,\bar B+\bar A\,\Delta b_w\big)\quad\text{(exact; the } w\text{-th term is writer } w\text{'s carrier share).}$$

A writer whose write is identical in both members of the pair has zero carriage however large its mass. This distinction was learned
the hard way in this line (§4.5) and is now a library function (`dod_units.carrier_split`).

## 3. The circuit

```mermaid
flowchart LR
  T["noun token ' traders' / ' trader'<br/>(embedding, position 1)"] --> M1["MLP 1 (block 1)<br/>41–62% carrier into the MLP-3 units"]
  M1 --> M3["MLP-3 units 3465, 493"]
  M1 --> M4["MLP-4 units 2434, 2633"]
  M3 --> M5["MLP-5 units 1036, 2403, 2538, 3274, 3034"]
  M4 --> M5
  M5 --> M6["MLP-6 units 2483, 2826, 4131"]
  H63["head 6.3<br/>(constant multiplier: 32% mass, 3.5% carriage)"] -.-> M6
  M6 --> M8["MLP-8 unit 829 = plural detector<br/>MLP-8 unit 953 = singular detector"]
  M8 -->|"Down[:,829] along +r, Down[:,953] along −r"| N["number state at the noun (and verb)"]
  N --> R96["head 9.6"] & R124["head 12.4"] & R151["head 15.1"] & R105["head 10.5"]
  R96 --> F["residual at the final token"]; R124 --> F; R151 --> F; R105 --> F
  F --> Lg["rms → W: logit(they) − logit(he)"]
  F -.->|"MLPs 9–17 respond: net +0.05 of a −1.6 change; MLP 17 +0.17"| Lg
```

In words: the noun token is turned into a number feature by MLP 1; MLPs 2–6 relay and sharpen it through a handful of named units per
block; MLP 8 holds two clean detectors, one that fires on plural nouns and one on singular nouns, writing opposite directions into the
residual; four heads read that state from the final position and write the ` they`−` he` direction. Head 6.3 sits in the chain as a gain,
not a carrier. Every arrow was measured, and every arrow's size under an edit is 2–4× smaller than its share in the fold (§4.6).

## 4. The math, stage by stage

### 4.1 The readers: four heads, weight-only direction [v76, v79, v134]

As in the person component, each head's readout direction is $\hat v_h=O_h^{\!\top}(u_{\text{they}}-u_{\text{he}})$ from the weights alone,
and removal is the projection of the head's slice onto $\hat v_h$ at the final query. The set {9.6, 12.4, 15.1, 10.5} removes 1.55 of the
2.05-logit margin (76%, positive on 96/96 rows); the sixteen norm-matched random directions remove at most 0.033; single heads remove 0.66,
0.51, 0.21, 0.18 logits, and the additivity gap is 0.015 against a bar of 0.046. Keeping only the four heads' projections retains 1.53 of the
1.55 (98%) [v76]. Sixteen random head quadruples remove at most 0.081 and none is live [v79]. The same heads carry the person family along
an *orthogonal* direction: projecting them onto the ` myself`−` yourself` direction removes nothing here [v134].

### 4.2 What the readers read: a computed state, not the token [v80]

The exact source fold of each coefficient $c_h=\hat v_h\cdot z_h(\text{final})$ splits it by source position and by value branch. Pooled
over the four heads, the noun position carries 34% and the token-only (block-0 value) branch **5%**; the largest share is "other" positions
(verb, object, connective: 40–57% per head), and the final token itself 12–20% for 12.4, 15.1 and 10.5. Compare the person heads: 61% pronoun
position, 38% token-only. These heads are contextual readers of a number state that has been written at the noun and copied along the sentence.

### 4.3 Where the state comes from: MLP 8 at unit grain [v168, v170, v169b]

Pooling MLP 8's write on 9.6's reader direction $r=V_{9.6}^{\!\top}\hat v$ at the noun over the 48 pairs, exactly by unit
($T_j=(r\cdot\text{Down}_{:,j})\,u_j$, closure $8\times10^{-5}$): the top 10 units carry 78%, and three carry most of that — **829**, **953**,
**1030** [v168]. Their weights say what they are [v170]: 829's value on 40 plural/singular noun pairs is large on the plural and near zero on
the singular (astronauts 764 vs astronaut 44; counselors 869 vs −16; 36/40 pairs), and its output column has cosine +0.63 with $r$; 953 is
the mirror (astronaut 579 vs astronauts −19; 37/40 pairs), output cosine −0.53. Zeroing the three at the noun in a plain forward removes
5.3% of the margin (positive 84/96 rows), 11.8% at all positions (the verb copy), against sixteen random 3-unit sets that remove at most
0.06%; the three unrelated readers move by less than the null [v169b]. Small and exactly sized: MLP 8 is about half of 9.6's noun read, 9.6
is 0.66 of the 1.55 set, and the three units are two thirds of MLP 8's part.

### 4.4 Inside the plural detector: mass, then carriage [v181, v188]

The pair fold of $u_{829}$ at the noun (26 writers: embedding, attention and MLP totals of blocks 0–7, the nine heads of block 8; closure
$10^{-7}$) has no dominant pair (largest 8%) and its *mass* is spread: pairs with MLP 6 carry 45%, with the embedding 26%, with block-7
attention 17%, with head 8.1 −1% [v181]. The *carrier* split of the same contrast reads differently: MLP 6 20%, **MLP 7 18%** (0.9% mass),
embedding 17%, MLP 5 16%, MLPs 3 and 4 6% each, **block-7 attention 3%** (17% mass), heads 8.1 and 8.8 under 4% [v188]. The MLP stack
carries two thirds of the number; the attention writes are multipliers.

### 4.5 One layer down, and the lesson [v182–v187, v191]

The exact leave-one-unit-out change of $u_{829}$ — $D_j=u(\hat x)-u(\hat x-c_j)$ with $c_j$ the MLP-6 unit's write as it reaches block 8 —
names three MLP-6 units, **2483, 2826, 4131**, carrying 57% of MLP 6's part [v182]. Their own pair folds put 32% of 2483's mass on head 6.3
[v183]; the source fold of that head at the noun found only a third from the noun position and two thirds from the sentence start [v185],
which I registered as a pattern-borne signal — falsified (pattern terms 1.8%) [v186]. The three-way split (pattern / value at a fixed reader /
reader) settled it: the start token's value cannot differ across the pair, so its share is **89% reader-borne** — the head writes the same
vector for plural and singular, and the contrast sits in the partner factor [v187]. At carrier level head 6.3 is 3.5% of 2483 and the MLP
stack 1–5 is 88% (2826: 78%) [v191]. Hence §2's rule: report mass and carriage both, and let edits size.

### 4.6 Down the stack: named units to MLP 3, computed from MLP 1 [v192–v194, v201]

Carrier-level unit censuses name the carriers at each block: MLP-5 units **1036, 2403, 2538, 3274, 3034** carry 64% of MLP 5's part of 2483
(top-10) [v192]; into unit 1036, MLP-3 units **3465, 493** (16%, 13%) and MLP-4 units **2434, 2633** (23%, 21%) [v194]; and the MLP-3 units are
themselves carried by **MLP 1** (41% and 62%) and MLP 2 (22%, 49%), with the embedding at 5–6% and head 3.5 at 11.5% for 493 [v201]. At no
stage is the raw token direction the carrier: the number feature is computed by the first MLP and relayed upward.

### 4.7 Edits decide: three stages, one calibration [v169b, v195, v196]

| edit at the noun (plain forward) | on the next unit | on the plural detector 829 | on the ` they`−` he` margin | random-set null |
|---|---|---|---|---|
| MLP-8 {829, 953, 1030} | — | — | −5.3% (−11.8% all positions) | ≤ 0.06% |
| MLP-6 {2483, 2826, 4131} | — | −6.3% | −0.67% | ≤ 0.22% / 0.02% |
| MLP-5 {1036, 2403, 2538, 3274, 3034} | 2483: −6.0% | −3.9% | −1.3% | ≤ 0.9% / 0.6% / 0.07% |

Every edit is real (6–29× its null) and shrinks rather than grows the signal; every one is 2–4× smaller than its first-order carrier share
(MLP 5's five were 22% of 2483 by carriage). Shares rank; edits size.

## 5. Out of the panel: natural text [v77, v78, v197–v200]

Natural rows: 64 FineWeb and 64 Pile snippets from an outcome-blind miner, 16 per (noun number, next pronoun) cell. The head set is live
on both sources (65% and 56% of the congruent margin) but fails the counter-case: removing it does not help the pronoun that disagrees
with the noun [v77, v78 pred_f]. The units do better:

| zeroed at the cue | congruent rows: margin lost | positive | incongruent rows: shift *toward* the label | null (16 random sets) |
|---|---|---|---|---|
| MLP-8 {829, 953, 1030} | 3.3% (7.3% all positions) | 77% | yes, −1.7% | ≤ 0.0005 logits |
| 829 alone / 953 alone | 1.9% / 0.9% | — | 829 acts only on plural-noun rows, 953 only on singular | — |
| MLP-6 {2483, 2826, 4131} | 0.8% | 72% | yes, −0.7% | ≤ 0.0007 |
| MLP-5 five | 0.498% (bar 0.5%: fails as written) | 61% | yes, −0.1% | ≤ 0.0003 |

The units carry the noun's number and nothing else, so on incongruent text removing them helps the pronoun the noun disagrees with. The
MLP-6 trio's all-positions edit flips sign (−0.16%): those units do other work away from the noun, a declared limit of the noun-position claim.

## 6. A worked example

"The **traders** lost the coin and so": at position 1, unit 829 of MLP 8 is around $+600$ on plural agent nouns and near zero on their
singulars (the vocabulary pairs above), so MLP 8 writes $\approx 600\,\text{Down}_{:,829}$, whose cosine with 9.6's reader direction is
$+0.63$; on "The **trader**" 829 is silent and 953 fires instead, writing along $-r$. Head 9.6 at the final token reads a mixture of that
state at the noun (48% of its coefficient) and its copy at the verb and connective (51%), with 5.5% of the total from the token-only branch; its slice
projected on $\hat v_{9.6}$ contributes 0.66 of the 2.05-logit margin, and the four heads together 1.55. Zero the three MLP-8 units at the noun
and the margin loses 0.11 logits; zero them everywhere and it loses 0.24; zero sixteen random triples and it moves by a thousandth.

## 7. Reproducing it

Rows: `run_pronoun_number_dod_battery_v76.build()`. The unit-grain chain, from `basis_aligned/bilinear_quotient/ops`:

```python
import aspectual_dod_lib as L, circuit_fast_screen_producer as producer, dod_units
import run_pronoun_number_dod_battery_v76 as g
rows, they, he, agents, objects = g.build()
nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
noun = lambda row: {"noun": next(i for i, t in enumerate(row.ids) if t in nouns)}
backend = producer.Bilin18TorchBackend.load("cuda"); fw = L.ManualForward(backend)
# leave-one-unit-out change of MLP-8 unit 829 for every MLP-6 unit (v182):
per_row, closure, n = dod_units.product_unit_census(backend, fw, rows, (6,), 8, 829, noun)
partner = {(r.construction, r.group, r.present): i for i, r in enumerate(rows)}
pooled = dod_units.pooled_contrast(rows, per_row[6], partner, "noun")      # -> 2483, 2826, 4131 on top
# carrier vs mass of one aligned pair, given per-writer factors aP, bP (plural) and aS, bS (singular) (v188):
ref, carrier, mass = dod_units.carrier_split(aP, bP, aS, bS)
```
Runners: `run_pronoun_number_dod_*_v{76,79,80,84,168,169b,170,181,182,183,185,186,187,188,191,192,193,194,195,196,197,198,199,200,201}.py`;
every one is enqueued through `ops/enqueue.sh` and writes the receipt named in brackets above.

## 8. What is and is not closed

Closed: the readers and their direction (weights only); the MLP-8 detectors as units with a vocabulary-level meaning; named carriers at MLPs
3–6; the computation bottoming out in MLP 1; OOD behaviour of the top three stages, including the counter-case. Open, and declared: the
MLP-1 computation itself — its unit census is spread (top-10 units carry 38%, none above 9%) and MLP 2 holds one large opposing unit, 2907,
against many small carriers [v202], so MLP 1 is a port at unit grain and the named chain starts at MLP 3; why edits run 2–4× below carrier shares (the later blocks' nonlinear compensation is the
candidate, not shown); the verb-position copy of the state (11.8% vs 5.3%) is the same two detectors re-firing at the verb (829 33%, 953 25% of MLP 8's write there [v203]), and at the verb the carriers are again the MLPs at that position (74%; attention blocks 4 and 5 lead the attention share at 9% and 8% [v204]) — the head that brings the feature to the verb is 4.5 (94% of block 4's contrast at the verb, 97% of it from the noun position, 26% token-only [v205]), with block 5's head 5.3 re-reading the verb position rather than copying. Zeroing 4.5 at the verb removes 7.7% of the margin, ten times any other block-4 head, but it is not selective (animacy and tense readers move five times their null [v206, v207b]): 4.5 is a shared subject-feature copier, as 8.1 is a shared token copier across families, so it is a carrier the number component uses, not a part of it. The number does ride a separable direction of that copy: removing only the component of 4.5's verb slice along $O_{4.5}^{\!\top} r$, with $r$ the plural detector's product gradient at the verb, takes 4.3% of the margin (55% of the whole-slice effect), beats sixteen random directions (≤ 0.6%) and leaves the animacy and tense readers inside their gates [v208] — a direction of a shared head is a component even when the head is not (with a norm-matched random null the picture holds: random directions ≤ 0.9%, readers at the null level [v209]; and a single fixed vector, the mean of the per-row directions, does 84% of the per-row job — 3.6% of the margin, selective, four times the matched null [v210] — so the number component at the verb is one 128-d direction of head 4.5; on natural text that fixed direction is weak, 0.8% of the congruent margin at twice the matched null, and behaves as a plural/they direction rather than a number contrast [v212] — the per-row directions are in fact one axis whose sign encodes the number (mean |cosine| 0.93; every singular row points opposite to every plural row [v213]), so the fixed axis is the component and its weakness out of the panel is real, not a construction artefact); unit 1030's role; and
the MLP-6 trio's off-noun behaviour. Failed predictions on the way are on the scorecard: 35 of 129 registered readings on this line's receipts were false.

### Pass over the draft (what I changed after rereading)

- The first draft called head 6.3 "part of the chain"; it is a gain. The diagram now draws it dashed and §4.5 tells the falsified nomination.
- Mass and carriage were used interchangeably in one paragraph of §4.4; §2 now defines both before either number appears, with the identity.
- The edit table in §4.7 originally listed shares next to edits without saying which is which; the last sentence now says "shares rank, edits size".
- The worked example quoted the vocabulary-level unit values as if they were the row's; the sentence now says "around" and cites the pairs.
- The natural-text table gained the MLP-5 row with its bar failure stated as a number, not as "narrowly missed".
