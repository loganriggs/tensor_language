# ODD_ATTENTION8H2_CORPUS_TRANSFER_V1 preregistration

The outcome-blind freezer selects one city occurrence per cached source document
by a fixed SHA256 ordering. It retains a contiguous32-token natural fragment,
with12 tokens before the city and19 after it, and creates exactly one
counterfactual arm by replacing that single city token with its frozen
cross-region counterpart. Eight contexts come from the untouched FineWeb
skip11000 cache and four from separately labelled Pile/reference documents.
All six previously frozen British/American spelling readouts are scored for each
context, giving144 rows and72 paired contrasts. Selection used no model scores,
activations, endpoint occurrence, or native capability.

The intervention keeps recipient head8.2 queries and applies the paired donor's
changed-city key/value source only to destinations after the city and before the
final readout token. It then propagates that write through block9 reentry/RMS and
the isolated head9.8-O current-value branch into the native suffix. Arms are
native, full-city, value-only, and inherited-city edge:576 body forwards. Report
FineWeb and Pile/reference separately; do not discard incapable pairs.

- **A — instrument:** frozen row hashes and one-token pairing hold; native/source
  replay, exact routing/value expansion, and outside-destination zeros are within
  `1e-5`; all intervention deltas are finite and live.
- **B — corpus capability:** at least half the paired city substitutions in each
  corpus give a positive native British-minus-American cue contrast, and the
  full-city paired-effect RMS is at least `1e-5`.
- **C — transported computation:** value-only/full-city paired-effect error is
  at most35%, routing complement misses by at least50%, and the inherited-city
  edge/full-city error is at most35% in each corpus.
- **D — signed manipulation:** the inherited-city edge moves at least75% of the
  capable paired contrasts toward the donor city's spelling convention, has
  cosine at least.6 with the full-city paired effect, and has mean absolute
  effect at least2% of the native cue magnitude in each corpus.
- **E — selectivity:** cat/dog, red/blue, Monday/Tuesday, and apple/orange RMS are
  each at most.5 of target-effect RMS in each corpus.

A failure of B makes this panel an honest natural-context capability null. A
failure of C or D rejects transfer of the current operational edge outside the
authored prompt family. This is cached-corpus transfer with one untouched arm,
not verified pretraining-disjoint OOD, a whole-head removal, or isolated model
compression. No thresholds or rows change after native scoring.
