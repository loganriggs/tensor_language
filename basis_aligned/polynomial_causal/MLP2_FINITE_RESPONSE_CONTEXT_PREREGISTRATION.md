# MLP2 finite-response context diagnostic — preregistration

## Status and scope

This is a post-validation exploratory diagnostic over the already-open 192-document
MLP2 VALIDATION role.  It cannot move the strict compression ledger, authorize a
replication role, or certify a replacement.  Its purpose is to falsify the next
mathematical interface cheaply: can the common finite-response mode discovered
across six equal-price MLP2 replacements be predicted from the state presented to
MLP2?

## Frozen objects

For document $d$ and arm $a$ in `SUFFIX`, `LOCAL`, `RMS`, `MASS`, `DERANGED`, and
`HASH_RANDOM`, define

$$
Y_{d,a}=\Delta\mathrm{CE}_{d,a}-\Delta\mathrm{CE}_{d,\mathrm{ZERO}}.
$$

Within each outer training fold, center and scale each arm of $Y$, then average the
six standardized values.  This average is the signed common partial-write response
target.  The outer test fold uses only the training fold's centering and scale.

The candidate input is the native pre-MLP2 RMS-normalized residual state $z_{d,t}$.
For each document, over scored positions only, form

$$
x_d=\left[\operatorname{mean}_t z_{d,t},
           \operatorname{mean}_t z_{d,t}^2\right]\in\mathbb{R}^{2304}.
$$

This is a deliberately small interface: first and diagonal-second state moments,
not token-level memorization or downstream logits.

## Frozen fit and split

- Use the 191 supported documents only.
- Make two outer folds by document-index parity; each document is predicted by a
  model that did not train on it.
- On each outer training fold, standardize features, fit PCA there only, and select
  rank $k\in\{4,8,16,32\}$ and ridge penalty
  $\lambda\in\{0.1,1,10,100\}$ by four-fold inner cross-validation.
- Refit the selected model on the outer training fold and predict the outer test
  fold.
- Compare against a token/count ridge baseline using scored fraction, unique-token
  fraction, adjacent-repeat fraction, earlier-repeat fraction, and the predefined
  copy-positive/repeat-negative/nonrepeat cell fractions.  Select its ridge penalty
  on the same inner folds.

## Frozen decision

The state interface is **promising** only if all of the following hold:

1. pooled out-of-fold Pearson correlation is at least 0.50;
2. both outer-fold Pearson correlations are positive;
3. its pooled squared error is at least 20% lower than the token/count baseline;
4. a label-permutation control has absolute pooled correlation below 0.20.

Otherwise it is not a sufficient gate at this aggregation level.  Failure prunes
document-level state moments, not tokenwise or nonlinear context gates.  Success
licenses a fresh FIT-trained finite intervention; it is not itself causal evidence.

Only aggregate metrics, hashes, model ranks, and penalties may be published.  Raw
states, targets, logits, or per-document predictions must not be written.
