# 18 September — Two readout families at the auxiliary slot (nine decisions, one recipe)

**Path now.** Every auxiliary decision in the corpus is read out by a small set of late heads writing along
`O_h^T(u_a − u_b)` at the final query. Across nine behaviours the same recipe (blind weight-only readout sweep of all
162 heads, then the top-4 set with a norm-matched random null and three unrelated readers) finds two families:
a **temporal family** {9.1, 9.4, 15.5} + 11.3 for tense/mood/aspect (four lines: aspectual has/had, temporal
will/had, narrative was/is, modal would/will) and a **number family** {5.7, 7.8, 9.7} + 11.3 for agreement (four
lines: perfect have/has, lexical were/was, quantifier was/were, coordination were/was). Head 11.3 is in both. A
non-auxiliary decision (preposition on/of) uses neither. **Delta since the last update:** the temporal and
narrative lines were taken through the full battery (fresh rows, frozen predictions, natural FineWeb/Pile rows,
keep-only, random-set nulls, composition); the number lines and the preposition contrast have the first two
rungs only (opened authored rows). Table and receipts: `basis_aligned/claude_hourly_review/SHARED_READOUT_COMPONENTS.md`.

**Most instructive results.** (1) Head 8.1 is one token-only temporal-cue reader (block-0 value of since/by or
tomorrow/earlier, written wherever its pattern lands) that feeds the temporal family on two lines; it is absent
where the cue is spread over a sentence (narrative, modal). (2) The strict four-way additivity bar fails on every
four-head set by 3–4% of the joint while all pairwise Möbius terms are below gate and the head split beats random
splits: the overshoot is a serial relay term (9.1 also writes the state 11.3 reads). (3) The MLP-generated parts of
the relay states did not close by folding on either line where tried (declared open ports; on the temporal line a
third of that part is the MLPs' response to 8.1's write).

**Metrics.** As in the aspectual report (damage, fraction of the native margin, norm-matched null max, retention);
fresh = rows unused by any selection step of that line; opened = the line's authored rows the sweep selected on.

## Status by line (five properties, head grain)

| line | Simple | Predicts OOD | Extracted | Selective | Composes |
|---|---|---|---|---|---|
| aspectual has/had | held | held (authored, FineWeb, Pile) | head boundary; 8.1 closed; 2 ports | held | held (5-way + random-split null) |
| temporal will/had | held | held (authored, FineWeb, Pile) | head boundary; 8.1 closed at the NP; 2 ports | held | pairwise held; strict 4-way fails 3% |
| narrative was/is | held | held (authored) | head boundary; ports open | held | pairwise held; strict 4-way fails 4% |
| modal would/will | sweep only | — | — | held on opened rows | — |
| number lines (4) | sweep only | — | — | gate needs a number-appropriate reader | — |

## Limitations
- Family membership for the modal and number lines rests on the line's authored rows (opened); the temporal and
  narrative lines show the sweep-selected sets survive fresh rows, but each new line needs its own confirmation.
- Was−were is a related reader for number decisions; the three-reader battery must swap it before a number-family
  selectivity claim.
- No natural-corpus rows for narrative, modal or number lines yet.

Receipts: `bilinear_quotient/circuits/followups/{aspectual_anchor,temporal_auxiliary,narrative_tense}_dod_*`,
`*_dod_reuse_census_v49b..v54`. Code: `ops/aspectual_dod_lib.py`, `ops/dod_reuse_census.py`, `ops/run_*_dod_*.py`.
