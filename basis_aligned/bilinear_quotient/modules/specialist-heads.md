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

## Open

- question_gates.py QUEUED: the template on 10.5 (targets = "?" in WH-opened sentences
  via §1313 QSTATE; key gate = WH openers, query gate = open clause). Assumption
  registered: annotator band = a02 by §1286-88; a band-arm miss localizes it elsewhere.
- Exclamation circuit (17.2+17.3): template not yet applied; L17 position means the
  "band" may be the whole stack below — expect the route to carry much more there.
- MDL bridge: price each specialist's kept description (patterns + route + head params at
  balanced gauge) against its capability nats — connects to modules/benchmark.md ladder.
