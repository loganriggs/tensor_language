# Rung 443 preregistration: can a structural score learned at MLP0 predict the MLP1 compiler bank?

Status: registered before reading numerical candidate metrics from the site1 ledger. CPU-only historical screen; no
model forward, GPU queue entry, candidate retraining, or prospective simplicity claim is authorized.

## Recovered old object

Compiler-v2.1 contains matching true-data and shuffled-data banks of108 candidate specifications at site0 (MLP0)
and site1 (MLP1). The candidates span five declared families: z-only affine Euclidean, state-complete affine
Euclidean, state-complete affine causal, state-complete native Euclidean, and state-complete native causal. The two
site ledgers were independently trained and frozen before their selectors ran.

This does not fill rung441's missing removal/composition outcomes. It tests one narrower promised consequence:
whether a structural rule learned on one circuit site predicts validation recovery at the next circuit site.

## Physical role separation

The `fit` phase may load only the site0 ledger. For every true-data candidate it reads declared family/interface/
grammar, rank or k, regularization value, total stored reals, float32 bits, and inference multiplies. The target is
site0 validation recovery. It fits a ridge-linear score on standardized features:

`log(price), log(operations), log(capacity), log1p(regularization), affine, state-complete, causal-metric`.

Ridge strength is selected from `{0, .01, .1, 1, 10, 100}` by leave-one-declared-family-out mean Spearman correlation
on site0 only. The fitted scaler, coefficients, chosen strength, feature order, candidate IDs, and site0-ledger hash
are frozen to a JSON file.

The `score` phase must verify that frozen hash before loading site1. It reports site1 true recovery and site1 shuffled
recovery but cannot refit or alter the score.

## Baselines and frozen measurements

- price baseline: larger `total_reals` predicts greater recovery;
- rank baseline: larger rank/k predicts greater recovery;
- direct-transfer baseline: site0 true recovery predicts site1 true recovery (reported as an upper, non-simplicity
  comparator, not a bar the learned score must beat);
- shuffle-learned control: the same frozen procedure fit to site0 shuffled recovery;
- 1,000 seeded label permutations provide a null Spearman distribution.

Pairwise accuracy is the fraction of all unequal site1 candidate pairs whose recovery ordering matches the score.
The predicted top decile is frozen by score before site1 is opened.

## Predictions and null

- A: exact108-ID equality across site0/site1 true/shuffle banks; receipt artifact hashes match; all price/capacity/
  operation fields are finite and positive; fit artifact is frozen before site1 load.
- B: learned-score site1 Spearman≥.50 and pairwise accuracy≥.70.
- C: learned-score Spearman exceeds both price and rank baselines by≥.10 and exceeds the shuffle-learned score by≥.15.
- D: in the frozen predicted top decile, median site1 true recovery exceeds median site1 shuffled recovery by≥.15.

Strong null: learned Spearman≤.20, learned score does not beat either price or rank, or the top-decile true-minus-
shuffle gap≤0. Under the null, structural definitions in this bank do not transfer even between adjacent early-MLP
sites. Otherwise this remains a historical/adaptive screen and only licenses designing a new prospective family;
it does not validate OOD, extraction, removal, composition, or semantics.

Literal price: CPU analysis only, zero deployed values and zero native calls. Claim level: historical statistical-
transfer screen.
