# Rung509 pre-outcome identifiability repair

Status: frozen after a synthetic ground-truth failure of the CPU fitting instrument, before any rung509 CUDA/model
outcome. The original free-response fit is preserved in commit `e3b2e9206`; its failure numbers are recorded in the
22:20 mathematical review.

## Failure

All six registered fits were run on an exact planted response tensor. They converged repeatably—minimum response
cosine`.990` and assignment cosine`.99996`—but recovered only two of eight planted response atoms; the other matched
cosines were between`-.115` and`.246`, with assignment MSE`.0613`. Thus restart and document-split agreement can
certify a repeatable optimizer convention rather than the planted variables.

## Repair

The eight response atoms are no longer free vectors. For discovery response rows `R[n]`, where `n` ranges over all
`4*253` score-source/exact-term observations, atom `k` is

`w[k] = sum_n alpha[k,n] R[n]`, with `alpha[k,n] >= 0` and `sum_n alpha[k,n]=1`.

The fit optimizes unconstrained logits followed by softmax to obtain `alpha`. Add an archetype-weight entropy penalty
`.01`, alongside the already frozen assignment-entropy penalty`.01`; all other optimizer settings remain fixed.

An atom is eligible only if its largest `alpha` is at least`.90`; the eight largest-weight observation indices must
be distinct; and the anchor observation identity must agree after atom matching across all independent seeds and the
two discovery halves. The existing response, assignment, support, and diversity gates remain.

Before CUDA, a deterministic synthetic separability test plants eight factorized source-assignment atoms with one
near-pure observed anchor each. Across the six independently initialized fits, every matched response atom must have
cosine at least`.90` with ground truth, every assignment tensor cosine at least`.80`, and every anchor identity must
be recovered. Failure blocks model execution and routes to the no-latent predictive-state quotient; it cannot be
repaired by using a shared initialization, selecting a favorable seed, changing atom count, or tuning penalties.

The real-data claim is narrower than generic dictionary identifiability: a passing atom is an observed-response-
anchored, source-factorized candidate that must still forecast held-out exact-term effects and pass physical removal
and pair composition. A real-data lack of anchors is a scientific null for this restricted dictionary, not permission
to relax `.90`.
