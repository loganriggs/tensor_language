# Hourly strategic review — 2026-08-29 07:40 UTC

## The update

The largest new scientific result is negative but clarifying: a rank-128 shared trunk
plus site-private output directions does **not** beat fully private maps at any of the
three rank-512-scale storage budgets. Sharing is useful only in the tight rank-64/128
regime measured earlier; at larger budgets it costs too many valuable private
directions. Separately, the deployed whole-program scale is now under direct test.
Multiplying all 36 compiled writes by 0.8 gave the best top-1 accuracy, while 0.5 gave
the strongest agreement with the native model. Scaling each site to its native live
norm was catastrophic. The missing all-position-CE sweep then completed: 0.8 was also
best by CE, but improved over deployed 1.0 by only 0.00407 / 0.00992 / 0.00447 nat.
Thus the deployed magnitude is already within 0.01 nat of optimal on every discovery
role, and two roles miss the prospective 0.005-nat materiality bar. The cheap scalar
diagnostic is closed; it is not the breakthrough or a reason to reinterpret the cost
frontier.

No strict whole-model explanation ledger moved. We learned which model classes fail,
but we did not complete a fresh extraction, selective removal, OOD transport, or
autonomous state interface.

## Terms and computations used in this update

- A **compiled write** is the residual-stream vector substituted for one attention or
  MLP module. There are 36 such sites: one attention and one MLP at each of 18 layers.
- A **global scale** (g) changes every compiled write (w_j) to (g w_j). This is
  one fitted degree of freedom, not 36. It is not a gauge symmetry: the unscaled token
  stream remains in each residual addition, so RMSNorm sees a different direction.
- **Top-1 accuracy** is the fraction of positions where the program's largest logit is
  the true next token.
- **Teacher agreement** is the fraction where program and native model choose the same
  top token. The reported enrichment divides this agreement by agreement after
  permuting predictions, so values such as (8\times) mean eight times the accidental
  agreement rate.
- **Cross-entropy** is

  \[
  \mathrm{CE}=-\frac1N\sum_{i=1}^{N}\log p_g(y_i\mid x_{\le i}),
  \]

  where (y_i) is the true next token. Lower is better. This is the model's training
  objective and the currency used by the recent compression frontier.
- **Teacher KL** is

  \[
  \mathrm{KL}(p_{\rm native}\Vert p_g)
  =\sum_v p_{\rm native}(v)
  \log\frac{p_{\rm native}(v)}{p_g(v)}.
  \]

  It measures distributional imitation rather than task accuracy. Low CE and low KL
  are related but need not select the same program.
- A **shared trunk plus private residuals** writes site (j)'s approximation as

  \[
  \widehat Y_j=X_jA_j^{(0)}V_0^\top+X_jA_j^{(p)}U_j^\top.
  \]

  (V_0) is one output basis shared by all sites; (U_j) contains directions unique
  to site (j). Literal storage is matched against an all-private endpoint. The
  hierarchy is useful only if sharing (V_0) buys more than the private rank slots it
  displaces.
- **Native Down** means retaining the original MLP matrix that maps selected product
  gates back into the residual stream. In Family F, it had worse local write error but
  better downstream KL than a refitted Down matrix. That disagreement is why the
  prospective behavioral-port experiment matters.

## What is actually explained

| Currency | Strict explained fraction | Meaning of the remainder |
|---|---:|---|
| Replaceable structural write interfaces | 36/36 | We can intercept every write, but do not thereby know its semantics or autonomous input state |
| Original storage with consequence-certified removal | 5.3481% | 94.6519% lacks a whole-program removal certificate |
| Named causal CE headroom recovered | 10.923% | 4.72714 nat, or 89.077%, remains unnamed |
| Terminal extraction/removal/OOD actions | 0/68 | No behavior-by-path cell has passed the full action suite |

The compiler has much broader **behavioral coverage** than these strict fractions: on
covered current tokens it tracks native top-1 choices about 7.2 times chance, and its
fallback tracks them about 2.9--3.6 times chance. That is evidence of a useful
approximate function, not yet a semantic decomposition or removal certificate.

## New interaction evidence

Restoring any one of the 18 native MLPs inside the covered context-free program was an
exact no-op. This is expected from the construction: the compiled prefix recreates the
same length-one stream used to build each positionwise MLP table. It does **not** mean
MLPs are irrelevant in the native contextual model.

Attention restorations are not additive. Late attention sites 13--17 carry much of the
remaining agreement, but restoring attention 5 or 6 alone can collapse performance to
chance. Their native writes are about 156 and 83 times the corresponding compiled-table
norms. Rescaling the restored write down to compiled magnitude largely removes the
collapse. Thus an isolated component can be harmful because it is inserted at the
wrong scale relative to the rest of the simplified program; leave-one-out scores do
not define independent circuit contributions.

The whole-program scale diagnostic found:

| Scale | Agreement enrichment | Program top-1 | All-position CE |
|---:|---|---|---|
| 0.50 | **8.27 / 8.41 / 8.91** | 13.47% / 13.82% / 13.41% | 6.18268 / 6.13618 / 6.17589 |
| 0.80 | 7.50 / 7.59 / 8.06 | **13.64% / 14.32% / 13.72%** | **6.00760 / 5.97485 / 5.99718** |
| 1.00 | 7.19 / 7.29 / 7.64 | 13.55% / 14.25% / 13.64% | 6.01167 / 5.98477 / 6.00165 |
| native-norm per site | 1.89 / 1.84 / 1.76 | 1.98% / 1.98% / 1.78% | 12.88014 / 12.93864 / 12.80992 |

The 0.5 arm agrees with the teacher more often but predicts the true token slightly
less often than 0.8. By CE it is fifth of the six uniform scales, so agreement and
task prediction disagree even more strongly in nats than in top-1. The per-site native-
norm arm is worse than a uniform predictor, whose CE would be
\(\log(50{,}304)\approx10.83\). The CE run is descriptive on already exposed roles;
it does not select a promotive scalar. A future scalar claim would still require a
calibration role, at least two genuinely fresh evaluation roles, document-level
uncertainty, and a separately reported KL selector. Given the small CE gain, only a
material fit-only KL signal now justifies that expense.

## Largest remaining gaps and blockers

1. **Autonomous state interface.** Native-stream rank-512 maps are good only when fed
   native upstream state. Recursively compiled state loses 1.09--1.27 nat, and fitting
   on that closed state was much worse. This remains the largest composability gap.
2. **Native-Down causal port.** The interesting KL/local-error reversal has no fresh
   finite-edit result. A registry-excluding 192-document freezer is now implemented
   and tested, but has not been independently audited or run, so no row receipt exists.
   The other blocker is a reviewed CUDA measurement adapter with source closure, call
   census, bootstrap, and receipt-last publication. GPU availability is not the blocker.
3. **Terminal copy circuit.** Prior data name six copy/induction heads and a strong
   four-head subset, but no fresh extraction/removal/OOD cell has run. The new contract
   has deterministic matched negatives and exact currencies, but still needs four
   fresh row roles, a per-head attention adapter, scorer/bootstrap authority, and a
   create-only result lifecycle. The contract is not an outcome.
4. **Finite transport composition.** The 384-document unique-row cache exists. The
   runner still lacks full gauge controls, finite nulls/inference, a matched PCA/RRR
   baseline, and terminal lifecycle closure. Rows alone are not an E3.2 result.
5. **OOD and selective action.** No current simplification has shown that a named
   feature can be extracted or removed on natural text and transported to a separately
   frozen OOD role without collateral damage.

## Candidate pruning and top five

The ordering below uses expected information gain, causal relevance, whole-model
composability, falsifiability, GPU cost, and redundancy with completed work.

1. **Fresh native-Down behavioral port.** Highest causal return: it directly tests the
   most confusing Family-F result with ordinary substitution and finite edits. It is
   presently blocked by fresh rows and the measurement adapter.
2. **One terminal copy/induction extraction-removal-OOD circuit.** Highest semantic and
   practical return, and it provides an external test of every simplicity measure. It
   is blocked by row/adaptor/scorer authorities.
3. **Fully controlled finite transport triangle.** It attacks the autonomous-state
   problem by predicting an unseen composition, but requires more control engineering
   than the first three and the infinitesimal response-panel predecessor was negative.
4. **One bounded tight-budget hierarchy.** Only the rank-64/128 price regime remains
   nonredundant. More rank-512 hierarchy points and attention/MLP typing are pruned by
   completed negative results.
5. **Conditional fresh KL scalar calibration.** CE is now too small and inconsistent
   with the 0.005-nat all-role bar to justify a full fresh CE experiment. Retain the
   one-degree-of-freedom grammar only if a fit-only KL scan shows material teacher-
   imitation gain; otherwise close it.

Also pruned: per-site live-norm scaling; more large-budget shared/private mixtures;
rotating a failed shared projector with an SAE before predictive directions are
retained; and interpreting leave-one-out restoration as an additive circuit score.

## Safe action executed this hour

- The all-position-CE global-scale sweep completed in 214.6 seconds. Scale 0.8 is best
  on all roles, but gains only 0.00407 / 0.00992 / 0.00447 nat over 1.0; the deployed
  scale stays within the registered 0.01-nat tolerance. Scale 0.5 is fifth by CE, and
  per-site native-norm scaling gives CE 12.81--12.94. This is a real discovery result,
  not an unrun runner, and it does not promote a scalar from exposed roles.
- The prospective scalar contract was independently red-teamed and hardened. It now
  requires at least two sealed roles, document-cluster simultaneous inference, exact
  capped-logit KL orientation, all-36-write folded replay, and separate CE-versus-KL
  claims. Its aggregate CPU contract intentionally cannot issue a promotive pass;
  8 focused tests pass.
- Native-Down and terminal-copy contracts were implemented and tested, but are listed
  as blockers rather than outcomes. No row, checkpoint, model, or GPU authority was
  opened by those scaffolds.
