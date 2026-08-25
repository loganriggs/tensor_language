# Specialist heads & the three-goal circuit loop (§1280-1328 arc)

The named single-head (and pair) specialists found by the concentration-screen method, and
where each stands on the user's three goals: **G1 extract** (pull the circuit out, run it
standalone), **G2 remove** (delete the important part, lose only the capability),
**G3 weights-read** (predict behavior from the weights, certify on natural text).
READ THIS BEFORE screening a new behavior or ablating any of these heads.

## The named inventory

| Circuit | Heads | Capability | Key numbers | §refs |
|---|---|---|---|---|
| Question | **10.5** | "?" after WH-openers; ALSO top comparative-refine carrier (§1345, drop +0.343 in the "than" kit — §1310 off-diagonal confirmed) | dmg 0.726 at "?", 5% at ".", 0.4% elsewhere; "question-specific" true at terminal grain only | §1284, §1313, §1345 |
| Comparative | **8.1** | "than" after non-adjacent comparative | 100.7% of L8's 1.64-nat damage; conc 276; criterion STREAM-computed (weights ratio 1.01); mark written by a02 band | §1303-1306 |
| Exclamation | **17.2 + 17.3** | "!" continuation | pair carries 91% (48.5/42.6); perfectly ADDITIVE half-heads (2.6%); criterion stream-computed | §1315-1320 |
| Stem matcher | **1.1 + 1.8** | inflection-variant induction | weights say STEM matcher; 78% of identical-support damage at 1569 natural variant positions (inside registered 30-80% band); control flat | §1307-1308 |
| Fetchers | 2.5, 3.8 (+8.3, 8.4 copy stations) | copy/induction fetch | see induction.md + §1207-1218 | — |
| Sink | 5.7 | position-0 constant | see attn-sink-5-7.md | — |

Circuit structural forms found so far: single OWNER (10.5, 8.1), CROWD (induction band),
additive PAIR (17.2/17.3), half-head pair (matchers). Criterion taxonomy (§1320, final):
all specialist criteria are STREAM-computed, not embedding-native — the front a02 band
writes the class marks the specialists fetch (§1286, §1306; the §1286 content/key
dissociation did NOT repeat for comparative, §1306 pred_c).

## Selectivity (G2 status, matrix-certified)

selectivity_matrix2.py (§1309-1310, disjoint masks, 1920 rows): the remove-circuit x
measure-behavior matrix is STRONGLY diagonal (diagonals 0.13-2.21, off-diag max 0.19) but
not absolute — replicated off-diagonal structure: matchers->successor 0.080,
question->than 0.171, fetchers->successor 0.037. Removal is surgical to first order;
quote the residue when claiming clean removal.

## Extraction ladder (G1 status)

| Rung | Grain | Ident recovery | Lesson | §ref |
|---|---|---|---|---|
| 1 | 7 circuit heads, mean rest | 19% | circuits = heads + upstream closure | §1311 |
| 2 | 33-head dependency closure | 30% | better, far from bar | §1312 |
| 3 | closure + **lambda-v1 route** through removed heads | **79%** | route grain is the lever; description length, not head count (user correction §1314) | §1316 |
| 4 | + printable stand-in code | axis transfers 38% | payload needs rms calibration | §1321 |
| 5 | rms-calibrated additive payload | negative | additive payload dead, 2 strikes | §1322 |

Route grain = keep each removed head's lambda*v1 term (block-0 broadcast, ~free in bits)
+ live patterns (window-foldable weights code, §1161-66), mean-replace fresh values only.
Shared-variable leak: the same broadcast carries the content pool, so elsewhere recovers
68% too (§1316 — expected, registered).

## Standing traps for this arc

- Small-n: comparative targets are rare (n=54 @ 960 rows; §1303-05 all flagged). Run 1920+.
- Additive payload injection into the stream is POISONED without per-position rms
  calibration (§1287 rule; §1321-22 two strikes).
- Concentration screens need BOTH controls (jitter + random) clean before a verdict
  (§1302 withheld; §1315 withheld; both later certified at more data).
- Label bugs: §1304's winner printed "10.1" for 8.1 (hardcoded L); check data rows.

## Comparative: G1+G2 measured (§1329)

comparative_extraction.py, route grain, n=110 targets: route alone recovers 43.7% of the
7.0-nat target gap (the shared-variable leak reaches specialist capabilities); route +
a02 band + 8.1 reaches 69.7% — but elsewhere-recovery 0.679 ~ target 0.697, so the kept
a02 band is the UNIVERSAL class-marker, not a comparative part. The two clean facts:
**8.1's within-extraction increment is 26x target-selective** (0.361 nats target vs 0.014
elsewhere — G2 verified surgically), and **8.1 without the band is worthless** (+0.025
over bare route) — the specialist is 100% conditional on its annotator, both directions
measured. G1 verdict: PARTIAL — 70% reached, kept description not capability-specific.

## Comparative rung 2: the annotator's service is QUERY-side (§1331)

Position-gating a02 at comparative (key-side) positions keeps only 27% of the band's
increment (+0.070 of +0.263 over route); a rank-1 mark direction carries ~55% of that
key-side piece (83-sample estimate, flagged). The dominant ~73% arrives query-side —
front-band processing at the prediction position that 8.1's q reads. BUT both gated
slices are PERFECTLY selective (elsewhere = route +-0.002): [route + comp-position
outputs + 8.1] is an honestly capability-specific 50%-extraction at zero elsewhere cost.
Next: comparative_query_side.py (key vs qry vs both gates; queued).

## Comparative circuit: CLOSED (§1333)

Final description: **[v1-route + a02 live inside two zero-bit token-defined windows
(comparative positions; positions with a comparative 2-20 back) + head 8.1]** = 0.659
target recovery — 85% of the whole-band arm's contribution — at elsewhere route+0.014.
Query-side dominates key-side 2.7:1 (+0.183 vs +0.068, mildly sub-additive at +0.222
joint). With §1329's removal results this circuit has: annotator windows, carrier head,
criterion sidedness, removal selectivity (26x), and description cost. First all-preds-TRUE
run of the thread. The band-arm anchor replicated 3x (0.694-0.697). Unchased tail: 15%
of band service from positions outside both windows.

**THE TEMPLATE** (for the next specialist): extraction at route grain + capability-window
gates on a02 + specialist head; arms full/ymean/route/band/key/qry/both; bars = band>=0.60,
qry-vs-key +-0.05, selectivity +-0.05 of route elsewhere.

## Post-template state (§1334-1345) — read CIRCUIT_REGISTRY.md for stages

- QUESTION closed at head grain (§1342): 16-head clause-gated kit + 10.5 = 0.641; the L4
  "crowd" is a TRIO by drop-cost (4.0/4.1/4.7) — add-grain masked it. Drop-grain is the
  kit instrument; add-grain only screens.
- COMPARATIVE closed 4-STAGE (§1344-45): annotate(a02, two zero-bit gates) -> fetch(8.1)
  -> refine({10.5, 12.8} + 11.7/11.6) -> readout = 0.76-0.78. Missing rest = generic
  mid-pool (L3-5, NOT kit material — elsewhere > target) + the UNPRICED MLP side
  (ATTENTION-ONLY convention: all 18 MLPs live in every extraction arm).
- ENTROPY-SETTER REFUTED for comparative (§1343, three ways: rank collapse = content
  missing; gain-freeze hurts; temperature s*=0.95). First gain-frozen leg of the template.
- CROSS-CIRCUIT MOONLIGHTING is standard, both directions: 0.3 + 1.1 serve the question
  kit (§1342); 10.5 is the TOP comparative-refine carrier (§1345 — §1310's off-diagonal
  seen causally). Kit costs across circuits are NOT additive; price shared heads once.
  Labels of the form "X-specific" hold at behavior grain, not service grain.
- 13.8 = **THE CLOSER** (§1356 rename): delimiter-general close-what's-open — owns every
  bracket subtype (§1341, shares 0.95-1.04) AND quote-closing (107.9% solo, neighbors
  net-negative). Kit: [route + depth>0 gate on a02 + 13.8] = 0.657 (§1346); a14 assist
  does not port (§1347); refine = L14-17 by construction (§1348). Union-kit test §1356+.

## Open

- Exclamation circuit (17.2+17.3): template not yet applied; L17 position — expect the
  route to carry much more.
- Capitalized: late-band shared (a15-17, no owner, §1339) — parked; band-gate extraction.
- The MLP side of every kit: unpriced. Bridge to modules/benchmark.md ladder (mlp1 table,
  top-MLP 16x16 context tables) is the exact-extraction frontier.
- Double-close "))" unevaluable on natural rows (n=4) — needs targeted corpus (§1341).

## CAPITALIZED COMMITTEE (S1397, extended S1411) — the anti-closer
TWELVE heads across a13-a17: {13.0, 13.5, 14.4, 14.6, 14.7, 15.3, 16.0, 16.4, 16.5,
17.0, 17.1, 17.2}, solo dCE .016-.035 each, NO owner anywhere (top shares .38-.55).
Sharp edge: next head below the .015 bar sits at .004. a14 is the band's biggest layer
(.089), missed by the S1339 screen (looked only at a15-17). 17.2 (expressive-broad)
leads = its 3rd moonlight. The committee is the named unit. REMOVAL HANDLE RE-PRICED (S1412): all-12 = .603
target damage (super-additive 1.5x parts; a13-14 and a15-17 crews are redundant
implementations — either alone ~.20); else spillover 6.7%. S1398's .21 undercounted 3x. Unit pricing (S1398): committee-only = 71% of band, else 2% (surgical); rest-20 unselective generic. REMOVAL GOAL DONE (7/162 heads). Grain (S1399): a16 trio conjunctive,
a17 trio payload-dominant. Kit (S1400-01): committee is a REMOVAL HANDLE ONLY — construction
marginal .008-.029 (redundant with commons+upstream); ungated commons serves capitalized .813.

## Two inventories, closed form (S1404)
Removal handles != construction components, now certified at the flagship: 13.8 (97%
bracket owner by ablation) has NEGATIVE drop cost in the unified 28-head kit — closer
targets are served without it under co-residence. Only 10.5 (question) is a load-bearing
specialist in construction (.195); 8.1 marginal .048. Individual drop costs do not
compose (redundancy = shared resource): 14 individually-cheap heads collapse the kit
when dropped together.

## NEWLINE-SETTER PAIR: 8.2 + 11.0 (S1415, from the head-grain damage mine)
8.2: newline-target damage +.1457, controls <= .0025 (cleanest first screen since the
closer). 11.0: +.0894, clean. Distinct from the old front-attention newline ROUTING
circuit (item 7) — these predict where newlines GO. 10.6 = quote helper +.036 clean
(possible open-side partner to 13.8's close ownership). Characterization: newline_pair_quote.
