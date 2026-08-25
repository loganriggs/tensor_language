# CIRCUIT REGISTRY — the one status board for circuit work

**Purpose: no duplicated work, no empty candidate pool.** Every circuit thread lives here
with its pipeline stage and §refs. CHECK THIS FILE (plus `modules/specialist-heads.md`
for mechanism detail) BEFORE starting any circuit thread; UPDATE IT in the same commit as
the ledger writeup that moves a circuit's stage. The older record systems it supersedes
as an index (they remain as data): CIRCUITS_INDEX.md (70 swarm-era leaf records),
CIRCUITS_SCOREBOARD.md (147 certified ownership clusters), circuits/*.json.

Dedup discipline (§1092): before opening a thread, grep the ledger with MULTIPLE
vocabularies (old arcs predate the § era: "sink", "cost map", dotted head names), check
this board, and check `leaf_duplicates.json` if drawing census-tag controls.

## The pipeline (stages a circuit moves through)

CANDIDATE → SCREENED (concentration + both controls clean, §1302/1315 standard)
→ OWNED (component named, certified at scale) → G3 weights-read (criterion read from
weights, certified on natural text) → G1 extracted (route grain + capability-window
gates + specialist; THE TEMPLATE, §1333, recipe in specialist-heads.md) → G2 removal
(surgical, selectivity quoted) → **CLOSED** (itemized description + both-direction
conditionality measured).

## Status board

| Circuit | Stage | Key facts | §refs |
|---|---|---|---|
| **Comparative (8.1)** | **CLOSED, 4-stage** | annotate(a02, gated)->fetch(8.1)->refine({10.5!, 12.8} +11.7/11.6)->readout = 0.76-0.78, else +0.01-0.03; 10.5 is the TOP refine carrier (+0.343 — §1310 off-diagonal confirmed causally); remaining gap = generic pool + unpriced MLP side | §1303-08, §1329-45 |
| **Question (10.5)** | **CLOSED at head grain** | slim kit = 16 heads + 10.5 = 0.641 (§1342); L4 crowd = trio 4.0/4.1/4.7; guests 0.3 + 1.1. NOTE §1345: 10.5 also = TOP comparative-refine carrier — "question-specific" label narrowed (terminal-grain true, service-grain shared) | §1284-§1342, §1345 |
| Exclamation (17.2+17.3) | RESTS: G1-specific blocked | v2 §1350: a05+pair 0.860 target (highest in program) BUT elsewhere 0.880 — generalist, no gate exists; depth matters +0.150 (criterion scope predicts annotator depth, 2v2 taxonomy); capability-specific kit blocked on backlog-#7 probe gate | §1315-20, §1349-50 |
| Stem matcher (1.1+1.8) | G3 done | weights-read certified on natural text (78% variant-support damage, in-band) | §1307-08 |
| Induction/copy stations | G1 partial | route-grain closure 79% (§1316); payload rungs 4-5 negative; stations 2.5/3.8/8.3/8.4 | §1311-16, §1204-18 |
| Sink (5.7) | CLOSED (pre-template) | one constant vector, mean-replacement free | sink arc, §1089-91 |
| L0H3 bigram router | CLOSED (pre-template) | exact bigram lookup table at zero cost | §1091, sink arc |
| Newline routing (a12) | OWNED, G1 open | §643-44: 80% collapse on front-attn ablation; low-rank removal fails (§652 — conditional computation) | §635-652, §513 |
| Delimiter, successor | OWNED (matrix row) | selectivity-matrix diagonals 0.41/0.35; no dedicated thread | §1309-10 |

## Candidate pool (behaviour-first screen, §513 — concentration measured, thread not opened)

| Behavior | Leader | Target dmg vs elsewhere | n | Note |
|---|---|---|---|---|
| **close_bracket** | **CLOSED, 4-stage** | kit 0.657 (§1346); a14 does NOT port (§1347 — ablation names roles, construction prices kits); refine = L14-17 by construction (ratio 2.4, +0.062, §1348) — downstream motif holds 2-for-2; generic-pool invariant +0.227==+0.225 across circuits | 1779 | thread rests; head-grain refine deprioritized (thin vs draw spread) |
| capitalized | a15-a17 BAND | +0.109/+0.098/+0.057 (a17/a16/a15), controls clean (§1339) | 29697 | NO layer-owner (1.1x) — first late-band shared capability; PARKED behind close-bracket; extraction starts from a band gate |
| ~~open_quote~~ | — | DEMOTED §1351: clean screen fails every bar (a10 +0.067, conc 2.7, jitter dirty); mask-sensitive (atlas n=90 vs clean n=1087); distributed-no-owner class | 1087 | negative, recorded |
| open_bracket | a17 | +0.070 vs +0.010 | 59 | shares a17 with capitalized — joint thread? |
| **digit** | **8.3 + 8.7 COMPLEMENTARY** | §1354: divides by EVIDENCE TYPE — 8.3 copy-flavored (2.9x per-position), 8.7 fresh-specialist NEGATIVE on copyable (interference inside a pair); moonlighting #4 | 9056 | thread rests fully mapped |
| sentence_end | a10 | +0.065 vs +0.020 | 482 | overlaps newline thread — dedup before opening |
| **quote_close** | a13 | atlas2 §1355: +0.475, ratio 30.5 — CLOSER-hypothesis test in flight (does 13.8 own it?) | — | generator-flagged |
| ellipsis | a17 | atlas2 §1355: +0.318, ratio 23.8 | — | pool, behind closer test |
| unit | a8 | atlas2 §1355: +0.278, ratio 7.5 (units follow numbers — a8 function context) | — | pool |
| ~~possessive, hyphen, ordinal, year~~ | — | fail generator bars §1355 (distributed / generalist profiles) | — | negative, recorded |
| ~~comma~~ | — | fails screen (distributed) | 476 | negative, recorded |
| ~~colon~~ | — | fails both controls | 74 | negative, recorded |

CAVEAT carried from §513: the atlas RATIO column used a biased elsewhere-denominator;
absolute damage pairs are quotable, ratios need the behaviour_atlas2 re-screen (vs global
mean). Any thread opened from this pool starts with its own §1302-standard screen anyway.

## Search heuristic (§1354): heads are FUNCTIONS, capabilities are CONTEXTS

To find a head's true function, intersect its capability appearances (0.3=bigram routing,
1.1=stem match, 10.5=terminal-state cashing, 8.3=copy — each shows up wherever its
function is useful). Capability kits share function-heads; price them once.

## Generators (when the pool runs low — run one, refill the pool)

1. **New class taxonomies through the behaviour screen** (one 136s run covers ~10
   classes, §513): unscreened candidates — possessive 's, hyphenation joins, ellipsis,
   list-item numerals, year/date completion, unit suffixes, pronoun case, subject-verb
   agreement s, quote-style matching (" vs '), URL/code tokens, ordinals.
2. **Damage-cluster mining** (CIRCUITS_SCOREBOARD.md, 147 certified clusters, 76%
   coverage): clusters with a dominant top component and no thread — e.g. 77 (attn13
   .202), 4 (attn1 .186), 25 (attn1 .168), 99 (mlp17 .159), 68 (mlp16 .146), 115
   (attn2 .143), 83 (attn8 .154). Pick, read its examples, name the behavior, screen it.
3. **Selectivity-matrix off-diagonals** (§1310): matchers→successor (0.080),
   question→than (0.171), fetchers→successor — each replicated residue = a candidate
   SHARED SUBROUTINE between two named circuits.
4. **The uncovered 24%**: well-predicted tokens in no certified slice (CIRCUITS.md
   coverage metric) — cluster their damage columns for new ownership candidates.
5. **BENCHMARK_BACKLOG.md** standing rungs (mode-conditioned gating #7 pairs naturally
   with the §1334 gating-inversion finding).

## Standing lessons for new threads (union; details in specialist-heads.md)

- Both controls (jitter + random) clean before any verdict; verdict withheld otherwise.
- n >= ~100 targets or run 1920+ rows; quote n always.
- Query-side gates beat key-side 2-3x on both circuits measured — design gates two-sided.
- Check gate DISJOINTNESS before registering a "both" arm (§1334 void-arm disclosure).
- Under partial extraction, MORE live components can be WORSE (§1334 inversion) — always
  run the gated arm even when the band arm exists.
- Route grain (§1316) is the extraction floor; additive payload injection is dead
  (§1321-22, two strikes).
