# Hourly strategic review — 2026-08-29 09:15 UTC

## Bottom line

The strict explained fraction has not increased. The concrete advance is that E4 now
has an independently reviewed, executable **scientific definition** of a causal copy
effect. This matters because the previous draft could have called observational
matching or a one-sided token swap a circuit. Two reviewers found three identity and
support defects; all were fixed, the combined suite passes `24/24`, and both reviewers
now give GO to the pure contract.

This is not an E4 evidence cell. No fresh row, threshold selection, checkpoint forward,
or behavior score was opened.

## Honest whole-model balance sheet

| Claim | Strict status | What it does and does not mean |
|---|---:|---|
| Structural interception | 36/36 sites | Every attention/MLP residual write can be intercepted; this is infrastructure, not semantics. |
| Consequence-certified storage removal | 5.3481% | A complete executable candidate passed its registered removal consequence. |
| Named causal CE | 10.923% | Named interventions explain this fraction in that ledger; 4.72714 nat remains unnamed. |
| Terminal extraction/removal/OOD actions | 0/68 | No terminal circuit has yet passed the full action contract. |
| Eight-hour entry points | E1 negative; E2 compression signal but semantic negatives; E3 tested states negative; E4 unmeasured | None is yet a predictive, editable whole-model language. |

The best positive structural fact remains exact covered-token MLP tables. On covered
current tokens, late live and table MLP writes agree to roughly
\(3.6\times10^{-7}\) relative error. Nearly all late-MLP prediction changes are on
uncovered current tokens, where the learned fallback differs from live computation.
Attention remains the contextual part: restoring attention 5 changes about 96% of both
covered and uncovered positions.

## What the corrected E4 computation means

At a scored position \(p\), let \(q=x_p\) be the current token and \(y=x_{p+1}\) the
target. Let \(j\) be the nearest earlier occurrence of \(q\). A natural copy-positive
position satisfies \(x_{j+1}=y\). A nearer contradictory successor defeats an older
matching occurrence.

For one attention intervention, the primary quantity is

\[
\tau_+=\operatorname{CE}_{\mathrm{ablated},+}
       -\operatorname{CE}_{\mathrm{native},+}.
\]

This compares two model arms on the exact same positive inputs. Positive \(\tau_+\)
means the intervention harmed copying. A matched negative cell gives \(\tau_-\), and

\[
S=\tau_+-\tau_-
\]

asks whether the harm is specific to copy contexts rather than generic damage. Matching
balances position, nearest-query distance, separate query/target fit frequencies, and
prior-query multiplicity; it is not treated as causality by itself.

The synthetic challenge compares the joint histories

\[
\{q\to y,r\to z\}\quad\hbox{and}\quad\{q\to z,r\to y\}
\]

with the same length, token multiset, current query and observed target. It measures
the change in \(\log p(y)-\log p(z)\). This is a clean joint association test, not an
isolated claim about a single bigram edge.

The physical attention intervention uses the tensor-additive head interface:

\[
w'=w_{\mathrm{full,native}}-\sum_{h\in H}w_h+\mu_H(p),
\]

where \(\mu_H(p)\) is a fit-only per-position mean. The full write and shared value bus
are checkpoint-bit-identical. The separate bfloat16 head sum has a measured
`0.002627--0.002667` relative accumulation-order residual, so an all-head integrity
control remains mandatory.

## Largest remaining gaps and confusing results

1. **No terminal behavior result.** The adapter and definition exist, but fresh rows,
   streaming scoring, physical dispatch and the receipt-last lifecycle are unfinished.
2. **No executable uncovered-token fallback.** The native-stream rank-512 map looked
   strong locally, but recursive closure cost `1.09--1.27` nat and self-consistent
   refitting was much worse. This is the largest MLP composition gap.
3. **No sufficient intermediate language.** The finite L8→L11→L14 rank-64 state had
   destination error `0.2709` and composed error `0.4520`; the representation itself
   was insufficient before composition.
4. **Native Down beats local refitting downstream.** For Family-F K512, native Down
   produced `0.05772` teacher KL while locally refitted Down produced `0.08476`, even
   though the refit improved local NRMSE. This is strong evidence that local MSE is the
   wrong simplicity objective, but the causal finite-edit port is not yet runnable.
5. **Compression has not produced semantic coordinates.** Shared low-rank output bases
   help at tight storage, but global/typed/hierarchical rank-512-scale variants do not
   beat independent maps enough to license selective edits.

## Candidate actions considered and pruned

- **Tensor structure:** use the exact additive attention-head writes for short terminal
  causal paths. Kept: it supports precise removal and composition at low GPU cost.
- **Polynomial structure:** preserve native MLP product gates and Down columns, then
  score finite suffix consequences. Kept, but second priority because its row and
  measurement lifecycle received a separate NO-GO audit.
- **Gauge/state structure:** fit a behavior-conditioned temporal state after a behavior
  is localized. Kept as a successor; a universal pointwise rank-64 state is pruned by
  E3.2.
- **Shared dictionaries/HOSVD:** only revisit at tight rank-64/128 storage or with a
  causal weighting. Large-budget shared/private and fixed-projector HOSVD forms are
  pruned by measured negatives.
- **Local MSE decoder refits:** pruned as a primary objective because Family F improved
  reconstruction while worsening downstream KL.
- **More closed-stream linear fitting:** pruned because direct recursive closure and
  self-consistent refitting both failed strongly.

## Top five actions now

1. **Finish the E4 outcome-blind row authority and token-only labels.** It is the
   remaining cheap prerequisite for the highest-information behavior screen and can be
   done CPU-side while another job owns the GPU. The freezer must fix tokenizer/source
   identity, code-register filtering, exact reload and failure publication before use.
2. **Implement E4 streaming scoring and physical selected-head dispatch.** This converts
   the now-sound definition into a causal result without retaining enormous full-logit
   tensors. It directly measures prediction, specificity, collateral CE and OOD.
3. **Repair and run the native-Down finite behavioral port.** This tests the strongest
   polynomial causal clue and can distinguish a real preserved program from downstream
   compensation.
4. **Fit a nonlinear uncovered-token fallback against downstream CE.** Preserve exact
   covered tables and target only the actual residual, using attention-conditioned
   quadratic or sparse-mixture grammars with native-free recursive deployment.
5. **Test tight-budget shared/private or causal-weighted rank allocation.** This is the
   surviving compression opportunity, but it ranks lower because compression without
   a selective causal API has not yet improved interpretation.

The ordering favors expected information gain, causal relevance, composability,
falsifiability and low marginal GPU cost, while rejecting branches already falsified by
whole-program tests.

## Action executed in this review

- Corrected nearest-successor labeling and separate zero-aware query/target frequencies.
- Replaced the old one-sided synthetic swap with a reciprocal association
  difference-in-differences across four position/distance templates.
- Added document-balanced retained-cell support and exact ordered-row support digests.
- Froze within-input causal CE and matched-negative specificity as separate quantities.
- Added the prospective attention-only/copy-only screening amendment and fixed a
  circular audit-commit dependency in the row freezer.
- Passed `24/24` focused tests and obtained independent GO from both the mathematical
  and artifact reviewers.

A separate `tracking_by_position.py` job currently uses approximately 18.4 GiB of GPU
memory. No competing GPU experiment was launched. Strict E4 status remains unmeasured.

