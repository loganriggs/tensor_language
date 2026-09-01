# Rung452 preregistration: uncertainty-separated MLP-PCA comparison freezer

Status: registered after rung451 froze the independent rows, but before those rows or their model outcomes are loaded
by this experiment. CPU-only analysis of rung450's original TEACHING outcomes.

## Claim and computation

Rung450 failed because the complete ranking of seven removal errors changed between two48-document halves. A complete
ranking treats a tiny difference exactly like a large one. This rung instead freezes which candidate pairs rung450
actually distinguished before looking at the new192-document role.

For every candidate and every original document, compute the numerator and denominator sums of squares used by the
registered normalized removal and composition errors. Draw2,000 bootstrap samples of96 documents with replacement,
using seed452 and the same sampled document indices for all candidates. For each of the21 candidate pairs, form the
bootstrap distribution of `error(left)-error(right)`. A pair is called uncertainty-separated only when its two-sided
95% percentile interval excludes zero. Freeze the candidate names, direction, interval, old full value, and bootstrap
mean and standard deviation. No threshold is tuned against the new role.

## Predictions and null

- **A:** exact rung450 result, three condition tensors, original-native tensor, candidate order, and source hashes hold.
- **B:** sufficient-statistic reconstruction matches all14 registered rung450 removal/composition full values to
  `1e-10`.
- **C:** deterministic shared document bootstrap yields13 uncertainty-separated removal pairs and16 composition pairs.
- **D:** the rung451 rows, new candidate outcomes, model, and SEALED_CONFIRMATION remain unopened.

**Strong null:** any source/identity/reconstruction clause fails, the deterministic pair counts differ, an output
would be overwritten, or any new/outcome role is accessed. Passing only freezes the questions for a separately
preregistered independent test. It does not repair rung450, count the family, or deploy values.
