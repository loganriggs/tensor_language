# Rung 440 preregistration: can the historical archive support a learned-simplicity test?

Status: registered before running the archive extractor. CPU-only schema and leakage audit; no model forward,
GPU queue entry, fitted predictor, or scientific promotion is authorized.

## Decision being made

The 20:54 plan proposed learning which definitions of simplicity predict four consequences: out-of-distribution
transport, circuit extraction, selective removal, and composition. A protocol document is not evidence that the
existing archive can support that experiment. Rung 440 asks the prior question: are there enough *comparable
candidate-program rows*, with pre-outcome structure and post-outcome consequences separable, to run a historical
held-family backtest without silently turning one experiment receipt into one candidate program?

This rung deliberately cannot conclude that a simplicity rule works. It can only license or block the historical
backtest.

## Frozen archive and canonicalization

- Search only top-level `basis_aligned/bilinear_quotient/*_results.json` files.
- Freeze integer rungs 300 through 436 inclusive. Rungs 437 onward were not available at the chosen historical
  cutoff and cannot enter either features or labels.
- A receipt is structurally eligible when it is a JSON object with an integer rung in that interval and at least
  three top-level Boolean keys beginning `pred_`.
- Files whose basename contains `invalid`, `diagnostic`, `initial`, `first_invalid`,
  `semantic_hash_repaired`, or `orthogonality_diagnostic`, or whose status contains `invalid`, are excluded with
  the reason retained.
- If more than one non-excluded receipt remains for a rung, that rung is marked ambiguous and excluded. The audit
  must not choose the most favorable receipt.
- Every included receipt and matched source file is SHA-256 hashed. Git's first-add timestamp supplies chronology;
  filesystem modification time is forbidden.

## Physical feature/label separation

The feature pass writes only:

- join key, rung, receipt/source hashes and repository-relative paths;
- first-add timestamp;
- mechanically classified module and grammar family;
- source abstract-syntax-tree counts;
- declared arm names/count;
- recursively flattened numeric values under keys containing `price`, `byte`, `scalar`, `parameter`, `rank`,
  `width`, `atom`, `edge`, `interface`, `operation`, or `runtime`.

It may not serialize predicate/null values, CE, loss, error, fidelity, accuracy, cosine, correlation, certificate,
damage, or other consequence metrics. The feature file is finalized and hashed before the label pass runs.

The label pass writes the join key, exact registered predicate/null Booleans, and a mechanical consequence-category
map. Instrument/configuration predicates remain marked as instruments rather than consequences. No structural
feature is copied into the label file.

## Frozen measurements and bars

`pred_a_separation_and_provenance` requires all included receipts to have hashes and Git timestamps, all ambiguous
or excluded rows to retain a reason, and a forbidden-key scan of the serialized feature file to find zero outcome
fields.

`pred_b_archive_volume` requires at least 80 canonical receipts, at least five module families, at least four
grammar families, source linkage for at least 90% of canonical receipts, and Git chronology for every receipt.

`pred_c_candidate_granularity_and_price` requires at least 70% of canonical receipts to expose an explicit arm map
and at least 70% to expose a machine-readable structural price. This is the crucial guard against pretending that
one multi-arm experiment is one candidate program.

`pred_d_consequence_coverage` requires at least three of {OOD/transport, extraction/identification,
removal/intervention, composition} to contain at least 25 registered Boolean labels each and span at least three
distinct module-by-grammar cells. Instrument predicates do not count.

Strong null: fewer than 50 canonical receipts, more than 20% of structurally eligible rungs ambiguous, or fewer
than two consequence categories pass their coverage bar. Under the strong null, abandon archive learning and
design a prospective bank from scratch.

## Frozen routing

- If A/B/C/D all pass, rung 441 may fit the already-specified chronological, whole-family held-out predictor and
  compare it with bytes, rank, sparse edges, and shuffled labels.
- If A/B pass but C or D fails, do **not** fit. Build a hand-reviewed candidate-arm manifest and a common
  consequence schema, then rerun this feasibility gate as a new generation.
- If A fails, repair only the extractor/provenance instrument without changing the archive cutoff or bars.
- A positive historical backtest remains an adaptive filter. It can only freeze a rule for a separately registered
  prospective family; it is not validation by itself.

Literal price: CPU archive analysis only; zero deployed model values and zero native-model calls. Claim level:
research-instrument feasibility.
